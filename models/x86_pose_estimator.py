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

    def post_process(self, raw_results):
        #TODO
        pass
