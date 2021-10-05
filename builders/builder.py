def build_detection_model(name, config):
    if name == "ssd":
        from detectors.x86_detector import X86Detector
        image_width = config["image_width"]
        image_height = config["image_height"]
        image_thresh = config["threshold"]
        detector = X86Detector(width=image_width, height=image_height, thresh=image_thresh)
    else:
        raise ValueError('Not supported detector named: ', name, ' for AlphaPose.')
    return detector
