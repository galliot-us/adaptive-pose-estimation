import numpy as np


def vectorized_boxes_to_centers_scales(boxes, aspect_ratio=1.0, scale_mult=1.25):
   """The vectorized version of convert box coordinates to center and scale."""
   centers = np.zeros((boxes.shape[0], 2), dtype=np.float32)
   w = boxes[:,2] - boxes[:,0]
   h = boxes[:,3] - boxes[:,1]
   centers[:,0] = boxes[:, 0] + w * 0.5
   centers[:,1] = boxes[:, 1] + h * 0.5
   idx = np.where(np.array((w > aspect_ratio * h), dtype=int) > 0)
   h[idx] = w[idx] / aspect_ratio
   idx = np.where(np.array((w < aspect_ratio * h), dtype=int) > 0 )
   w[idx] = h[idx] * aspect_ratio
   scales = np.zeros((boxes.shape[0], 2), dtype=np.float32)
   scales[:,0] = w
   scales[:,1] = h
   idx = np.where(centers[:,0] != -1)
   scales[idx,:] = scales[idx,:] * scale_mult
   return centers, scales

def vectorized_centers_scales_to_boxes(centers, scales):
    xmin = np.array(centers[:,0] - scales[:,0] * 0.5)
    ymin = np.array(centers[:,1] - scales[:,1] * 0.5)
    cropped_boxes = np.array([xmin, ymin,np.array(xmin + scales[:,0]),
                np.array(ymin + scales[:,1])])
    cropped_boxes = np.transpose(cropped_boxes)
    return cropped_boxes


def box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0, h * 1.0], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def center_scale_to_box(center, scale):
    w = scale[0]
    h = scale[1]
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox
