from .base_pose_estimator import BasePoseEstimator
from adaptive-object-detecton.detectors.x86_detector import X86Detector


class X86PoseEstimator(BasePoseEstimator):
    def __init__(self,
            detector_thresh
            pose_input_size=(256, 192),
            heatmap_size=(64, 48),
            ):

        super().__init__(detector_thresh)
        self.pose_input_size = pose_input_size
        self.heatmap_size = heatmap_size

    def load_model(self, detector_path, detector_label_map):
        self.detector = X86Detector(width=self.detector_width, height=self.detector_height, thresh=self.detector_thresh)
        self.detector.load_model(detector_path, detector_label_map)

        base_dir = "models/data/"
        model_name = "fastpose_tf"
        base_url = "https://raw.githubusercontent.com/neuralet/models/master/amd64/fastpose_tf/"
        model_file = model_name + ".tar.gz"
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            logging.info('model does not exist under: {}, downloading from {}'.format(str(model_path), base_url + model_file))
            os.makedirs(base_dir, exist_ok=True)
            wget.download(base_url + model_file, base_dir)
            with tarfile.open(base_dir + model_file, "r") as tar:
                tar.extractall(path=base_dir)
        model_dir = pathlib.Path(model_path) / "saved_model"
        model = tf.saved_model.load(str(model_dir))
        infr = model.signatures['serving_default']
        self.pose_model = infr


    def preprocess(self, raw_image):
        return self.detector.preprocess(raw_image)


    def inference(self, preprocessed_image):
        
        raw_detections = self.detector.inference(preprocessed_image)
        detections = prepare_detection_results(raw_detections, self.detector_width, self.detector_height)
        inps, cropped_boxes, boxes, scores, ids = transform_detections(inp_rgb, detections, (256, 192))
        raw_output = self.pose_model(inps)
        hm = raw_output['conv_out'].numpy()
        return (hm, cropped_boxes, boxes, scores, ids)

    def post_process(self, hm, cropped_boxes, boxes, scores, ids):
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
		preds_scores[k] = preds_scores[k][:,0].reshape([17,1])
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

    @staticmethod
    def heatmap_to_coord(hms, bbox, hms_flip=None, **kwargs):
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

    @staticmethod
    def transform_preds(coords, center, scale, output_size):
	target_coords = np.zeros(coords.shape)
	trans = get_affine_transform(center, scale, 0, output_size, inv=1)
	target_coords[0:2] = self.affine_transform(coords[0:2], trans)
	return target_coords

    @staticmethod
    def affine_transform(pt, t):
	new_pt = np.array([pt[0], pt[1], 1.]).T
	new_pt = np.dot(t, new_pt)
	return new_pt[:2]
