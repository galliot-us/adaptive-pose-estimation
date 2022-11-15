import os
import logging
import wget
import tarfile
import pathlib
import tensorflow as tf
import numpy as np
from .base_pose_estimator import BasePoseEstimator
import cv2
from adaptive_object_detection.detectors.x86_detector import X86Detector
from tools.convert_results_format import prepare_detection_results
from tools.bbox import box_to_center_scale, center_scale_to_box
from tools.transformations import get_affine_transform, im_to_tensor
from tools.pose_nms import pose_nms
from tools.transformations import get_affine_transform, get_max_pred


class X86PoseEstimator(BasePoseEstimator):
    def __init__(self,
                 detector_thresh,
                 detector_input_size=(300, 300),
                 pose_input_size=(256, 192),
                 heatmap_size=(64, 48),
                 ):

        super().__init__(detector_thresh)
        self.pose_input_size = pose_input_size
        self.heatmap_size = heatmap_size
        self.detector_height, self.detector_width = detector_input_size

    def load_model(self, detector_path, detector_label_map):
        self.detector = X86Detector(width=self.detector_width, height=self.detector_height, thresh=self.detector_thresh)
        self.detector.load_model(detector_path, detector_label_map)

        base_dir = "models/data/"
        model_name = "fastpose_tf"
        base_url = "https://media.githubusercontent.com/media/neuralet/models/master/amd64/fastpose_tf/"
        model_file = model_name + ".tar.gz"
        model_path = os.path.join(base_dir, model_file)
        if not os.path.isfile(model_path):
            logging.info(
                'model does not exist under: {}, downloading from {}'.format(str(model_path), base_url + model_file))
            os.makedirs(base_dir, exist_ok=True)
            wget.download(base_url + model_file, base_dir)
            with tarfile.open(base_dir + model_file, "r") as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=base_dir)
        model_dir = pathlib.Path(base_dir) / "saved_model"
        self.model = tf.saved_model.load(str(model_dir))
        self.pose_model = self.model.signatures['serving_default']

    def preprocess(self, raw_image):
        return self.detector.preprocess(raw_image)

    def inference(self, preprocessed_image):
        raw_detections = self.detector.inference(preprocessed_image)
        detections = prepare_detection_results(raw_detections, self.detector_width, self.detector_height)
        inps, cropped_boxes, boxes, scores, ids = self.transform_detections(preprocessed_image, detections)
        if inps.shape[0] == 0:
            return (None, None, None, None, None)
        raw_output = self.pose_model(inps)
        hm = raw_output['conv_out'].numpy()
        return (hm, cropped_boxes, boxes, scores, ids)

    def post_process(self, hm, cropped_boxes, boxes, scores, ids):
        if hm is None:
            return
        assert hm.ndim == 4
        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            hm_size = self.heatmap_size
            eval_joints = list(range(17))
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = self.heatmap_to_coord(hm[i, :, :, eval_joints], bbox, hm_shape=hm_size,
                                                           norm_type=None)

            pose_coords.append(pose_coord)
            pose_scores.append(pose_score)

        preds_img = np.array(pose_coords)
        preds_scores = np.array(pose_scores)

        #boxes, scores, ids, preds_img, preds_scores, pick_ids = \
        #    pose_nms(boxes, scores, ids, preds_img, preds_scores, 0)
        _result = []
        for k in range(len(scores)):
            if np.ndim(preds_scores[k] == 2):
                preds_scores[k] = preds_scores[k][:, 0].reshape([17, 1])
                _result.append(
                    {
                        'keypoints': preds_img[k],
                        'kp_score': preds_scores[k],
                        'proposal_score': np.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx': ids[k],
                        'bbox': [boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3]]
                    }
                )
        return _result

    def transform_detections(self, image, dets):
        input_size = self.pose_input_size
        if isinstance(dets, int):
            return 0, 0
        dets = dets[dets[:, 0] == 0]
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        ids = np.zeros(scores.shape)
        inps = np.zeros([boxes.shape[0], int(input_size[0]), int(input_size[1]), 3])
        cropped_boxes = np.zeros([boxes.shape[0], 4])
        image = image / 255.0
        image[..., 0] = image[..., 0] - 0.406
        image[..., 1] = image[..., 1] - 0.457
        image[..., 2] = image[..., 2] - 0.480
        aspect_ratio = input_size[1] / input_size[0]
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transform_single_detection(image, box, input_size, aspect_ratio)
            cropped_boxes[i] = np.float32(cropped_box)
        inps = im_to_tensor(inps)
        return inps, cropped_boxes, boxes, scores, ids

    @staticmethod
    def transform_single_detection(image, bbox, input_size, aspect_ratio):
        xmin, ymin, xmax, ymax = bbox
        center, scale = box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
        trans = get_affine_transform(center, scale, 0, [input_size[1], input_size[0]])
        img = cv2.warpAffine(image, trans, (int(input_size[1]), int(input_size[0])), flags=cv2.INTER_LINEAR)
        bbox = center_scale_to_box(center, scale)
        return img, bbox

    def heatmap_to_coord(self, hms, bbox, hms_flip=None, **kwargs):
        if hms_flip is not None:
            hms = (hms + hms_flip) / 2
        if not isinstance(hms, np.ndarray):
            hms = hms.cpu().data.numpy()
        coords, maxvals = get_max_pred(hms)

        hm_h = hms.shape[1]
        hm_w = hms.shape[2]

        # post-processing
        for p in range(coords.shape[0]):
            hm = hms[p]
            px = int(round(float(coords[p][0])))
            py = int(round(float(coords[p][1])))
            if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
                diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]))
                coords[p] += np.sign(diff) * .25

        preds = np.zeros_like(coords)

        # transform bbox to scale
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        center = np.array([xmin + w * 0.5, ymin + h * 0.5])
        scale = np.array([w, h])
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = self.transform_preds(coords[i], center, scale,
                                            [hm_w, hm_h])

        return preds, maxvals

    def transform_preds(self, coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = get_affine_transform(center, scale, 0, output_size, inv=1)
        target_coords[0:2] = self.affine_transform(coords[0:2], trans)
        return target_coords

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]
