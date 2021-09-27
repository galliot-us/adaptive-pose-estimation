import time
import cv2 as cv
import numpy as np
import argparse
import logging
import os
from adaptive_object_detection.utils.parse_label_map import create_category_index_dict
from tools.visualization_utils import visualize_poses

logging.basicConfig(level=logging.INFO)


def inference(args):
    device = args.device
    detector_input_width = args.detector_input_width
    detector_input_height = args.detector_input_height
    detector_thresh = args.detector_threshold
    pose_input_width = args.pose_input_width
    pose_input_height = args.pose_input_height
    heatmap_width = args.heatmap_width
    heatmap_height = args.heatmap_height
    label_map_file = args.label_map
    batch_size = args.batch_size
    pose_model_path = args.pose_model_path
    if not label_map_file:
        label_map_file = "adaptive_object_detection/utils/mscoco_label_map.pbtxt"
    label_map = create_category_index_dict(label_map_file)
    if device == "x86":
        from models.x86_pose_estimator import X86PoseEstimator
        pose_estimator = X86PoseEstimator(detector_thresh=detector_thresh,
                                          detector_input_size=(detector_input_height, detector_input_width),
                                          pose_input_size=(pose_input_height, pose_input_width),
                                          heatmap_size=(heatmap_height, heatmap_width))
    elif device == "jetson":
        from models.jetson_pose_estimator import TRTPoseEstimator
        pose_estimator = TRTPoseEstimator(detector_thresh=detector_thresh,
                                          detector_input_size=(detector_input_height, detector_input_width),
                                          pose_input_size=(pose_input_height, pose_input_width),
                                          heatmap_size=(heatmap_height, heatmap_width),
                                          batch_size=batch_size,
                                          pose_model_path=pose_model_path)
    else:
        raise ValueError("device should be 'x86' or 'jetson' but you provided {0}".format(device))
    video_uri = args.input_video
    if not os.path.isfile(video_uri):
        raise FileNotFoundError('video file does not exist under: {}'.format(video_uri))
    if not os.path.isdir(args.out_dir):
        logging.info("the provided output directory : {0} is not exist".format(args.out_dir))
        logging.info("creating output directory : {0}".format(args.out_dir))
        os.makedirs(args.out_dir, exist_ok=True)

    file_name = ".".join((video_uri.split("/")[-1]).split(".")[:-1])
    input_cap = cv.VideoCapture(video_uri)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_cap = cv.VideoWriter(os.path.join(args.out_dir, file_name + "_neuralet_pose.avi"), fourcc, 25,
                             (args.out_width, args.out_height))
    if (input_cap.isOpened()):
        print('opened video ', video_uri)
    else:
        print('failed to load video ', video_uri)
        return
    pose_estimator.load_model(args.detector_model_path, label_map)
    running_video = True
    frame_number = 0
    start_time = time.perf_counter()
    times = []
    while input_cap.isOpened() and running_video:
        ret, cv_image = input_cap.read()
        if not ret:
            running_video = False
        if np.shape(cv_image) != ():
            out_frame = cv.resize(cv_image, (args.out_width, args.out_height))
            preprocessed_image = pose_estimator.preprocess(cv_image)
            start_time = time.perf_counter()
            result_raw = pose_estimator.inference(preprocessed_image)
            finish_time = time.perf_counter()
            result = pose_estimator.post_process(*result_raw)
            times.append(round(finish_time - start_time, 6))
            if result is not None:
                out_frame = visualize_poses(out_frame, result, (detector_input_width, detector_input_height))
            out_cap.write(out_frame)
            frame_number += 1
            if frame_number % 100 == 0:
                average_fps = round(1 / (sum(times) / len(times)), 2)
                times = []
                logging.info("processed {0} frames, average FPS: {1}".format(frame_number, average_fps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script runs the inference of pose estimation models")
    parser.add_argument("--device", type=str, default="x86",
                        help="supports x86 and jetson device")
    parser.add_argument("--input_video", type=str, required=True, help="input video path")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to store output video")
    parser.add_argument("--detector_model_path", type=str,
                        help="path to the detector model files, if not provided the default COCO models will be used")
    parser.add_argument("--label_map", type=str, help="path to the label map file")
    parser.add_argument("--detector_threshold", type=float, default=0.1, help="detection's score threshold")
    parser.add_argument("--detector_input_width", type=int, default=300, help="width of the detector's input")
    parser.add_argument("--detector_input_height", type=int, default=300, help="height of the detector's input")
    parser.add_argument("--pose_input_width", type=int, default=192, help="width of the pose estimator's input")
    parser.add_argument("--pose_input_height", type=int, default=256, help="height of the pose estomator's input")
    parser.add_argument("--heatmap_width", type=int, default=48, help="width of the pose haetmap")
    parser.add_argument("--heatmap_height", type=int, default=64, help="height of the pose heatmap")
    parser.add_argument("--out_width", type=int, default=960, help="width of the output video")
    parser.add_argument("--out_height", type=int, default=540, help="height of the output video")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size of pose estimator (it only works for jetson)")
    parser.add_argument("--pose_model_path", type=str, default="models/data/fast_pose_fp16_b8.trt",
                        help="using for jetson, path to the pose estimator model file, if not provided the default the model is loaded by default path")

    args = parser.parse_args()

    if (vars(args)["detector_model_path"]) and (not vars(args)["label_map"]):
        parser.error('If you pass model_path you should pass label_map too')

    inference(args)
