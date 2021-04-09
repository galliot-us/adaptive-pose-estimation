import abc


class BasePoseEstimator(abc.ABC):
    """
    A base class for detectors.
    The following should be overridden:
    load_model()
    preprocess()
    inference()
    """
    def __init__(self, detector_thresh):
        self.detector_thresh = detector_thresh
        self.pose_model = None
        self.detection_model = None

    @abc.abstractmethod
    def load_model(self, model_path):
        raise NotImplementedError

 
    @abc.abstractmethod
    def preprocess(self, raw_image):
        raise NotImplementedError

 
    @abc.abstractmethod
    def inference(self, preprocessed_image):
        raise NotImplementedError

    
    @abc.abstractmethod
    def post_process(self, raw_results):
        raise NotImplementedError
