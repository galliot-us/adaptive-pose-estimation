from .bbox import box_to_center_scale, center_scale_to_box
from .transformations import get_affine_transform, im_to_tensor
import numpy as np
import cv2
import tensorflow as tf

def transform_detections(image, dets, input_size):
    if isinstance(dets, int):
        return 0, 0
    dets = dets[dets[:, 0] == 0]
    boxes = dets[:, 1:5]
    scores = dets[:, 5:6]
    ids = np.zeros(scores.shape)
    inps = np.zeros([boxes.shape[0], int(input_size[0]), int(input_size[1]), 3])
    cropped_boxes = np.zeros([boxes.shape[0], 4])
    for i, box in enumerate(boxes):
        inps[i], cropped_box = transform_single_detection(image, box, input_size)
        cropped_boxes[i] = np.float32(cropped_box)
    inps = im_to_tensor(inps) 
    return inps, cropped_boxes, boxes, scores, ids


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
    #img = im_to_tensor(img)
    return img, bbox
