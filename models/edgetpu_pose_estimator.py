import logging
from adaptive_object_detection.detectors.edgetpu_detector import EdgeTpuDetector
from .base_pose_estimator import BasePoseEstimator
import os
import wget
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter
from tools.pose_nms import pose_nms
from tools.bbox import box_to_center_scale, center_scale_to_box
from tools.transformations import get_affine_transform, get_max_pred
from tools.convert_results_format import prepare_detection_results
import numpy as np


class EdgeTPUPoseEstimator(BasePoseEstimator):
    def __init__(self,
                 detector_thresh,
                 detector_input_size=(300, 300),
                 pose_input_size=(256, 192),
                 heatmap_size=(64, 48),
                 batch_size=8,
                 pose_model_path=None
                 ):
        super().__init__(detector_thresh)
        self.detector_height, self.detector_width = detector_input_size
        self.pose_input_size = pose_input_size
        self.heatmap_size = heatmap_size
        self.batch_size = batch_size
        self.pose_model_path = pose_model_path

        self.model = None
        self.detector = None
        self.raw_frame = None

        self.input_details = None
        self.output_details = None

    def load_model(self, detector_path, detector_label_map):
        # Load detector model
        self.detector = EdgeTpuDetector(width=self.detector_width, height=self.detector_height,
                                        thresh=self.detector_thresh)
        self.detector.load_model(detector_path, detector_label_map)

        # Load pose estimator model
        if not self.pose_model_path:
            logging.info("you didn't specify the model file so the Neuralet fastpose pretrained model will be used")
            base_url = None  # TODO
            model_file = '.tflite'  # TODO
            base_dir = "detectors/data"
            model_path = os.path.join(base_dir, model_file)
            if not os.path.isfile(model_path):
                logging.info('model does not exist under: {}, downloading from {}'.format(str(model_path),
                                                                                          base_url + model_file))
                os.makedirs(base_dir, exist_ok=True)
                wget.download(base_url + model_file, model_path)
        self.model = Interpreter(model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
        self.model.allocate_tensors()
        # Get the model input and output tensor details
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def preprocess(self, raw_image):
        self.raw_frame = raw_image
        return self.detector.preprocess(raw_image)

    def inference(self, preprocessed_image):
        raw_detections = self.detector.inference(preprocessed_image)
        detections = prepare_detection_results(raw_detections, self.detector_width, self.detector_height)
        resized_pose_img = cv2.resize(self.raw_frame, (self.detector_width, self.detector_height))
        rgb_resized_img = cv2.cvtColor(resized_pose_img, cv2.COLOR_BGR2RGB)
        inps, cropped_boxes, boxes, scores, ids = self.transform_detections(rgb_resized_img, detections)
        if inps.shape[0] == 0:
            return (None, None, None, None, None)

        if not self.model:
            raise RuntimeError("first load the model with 'load_model()' method then call inferece()")
        input_image = np.expand_dims(inps, axis=0)  # TODO for batch

        self.model.set_tensor(self.input_details, input_image)  # TODO
        self.model.invoke()
        result = self.model.get_tensor(self.output_details)
        return (result, cropped_boxes, boxes, scores, ids)

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
        boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            pose_nms(boxes, scores, ids, preds_img, preds_scores, 0)
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
        # image = image.transpose(2,1,0)
        input_size = self.pose_input_size
        if isinstance(dets, int):
            return 0, 0
        dets = dets[dets[:, 0] == 0]
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        ids = np.zeros(scores.shape)
        inps = np.zeros([boxes.shape[0], int(input_size[0]), int(input_size[1]), 3])
        cropped_boxes = np.zeros([boxes.shape[0], 4])
        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transform_single_detection(image, box, input_size)
            cropped_boxes[i] = np.float32(cropped_box)
        # inps = im_to_tensor(inps)
        return inps, cropped_boxes, boxes, scores, ids

    @staticmethod
    def transform_single_detection(image, bbox, input_size):
        aspect_ratio = input_size[1] / input_size[0]
        xmin, ymin, xmax, ymax = bbox
        center, scale = box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
        scale = scale * 1.0

        input_size = input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        inp_h, inp_w = input_size
        img = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = center_scale_to_box(center, scale)
        img = img / 255.0
        img[..., 0] = img[..., 0] - 0.406
        img[..., 1] = img[..., 1] - 0.457
        img[..., 2] = img[..., 2] - 0.480
        # img = im_to_tensor(img)
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
    def transform_single_detection(image, bbox, input_size):
        aspect_ratio = input_size[1] / input_size[0]
        xmin, ymin, xmax, ymax = bbox
        center, scale = box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
        scale = scale * 1.0

        input_size = input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        inp_h, inp_w = input_size
        img = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = center_scale_to_box(center, scale)
        img = img / 255.0
        img[..., 0] = img[..., 0] - 0.406
        img[..., 1] = img[..., 1] - 0.457
        img[..., 2] = img[..., 2] - 0.480
        # img = im_to_tensor(img)
        return img, bbox

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]
