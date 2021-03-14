import numpy as np
from utils.pose_nms import pose_nms
from utils.transformations import get_affine_transform, get_max_pred


def post_process(hm, cropped_boxes, boxes, scores, ids):
    assert hm.ndim == 4
    pose_coords = []
    pose_scores = []
    for i in range(hm.shape[0]):
        hm_size = (64, 48)  # TODO: from cfg
        eval_joints = list(range(17))  # TODO: from cfg
        bbox = cropped_boxes[i].tolist()
        pose_coord, pose_score = heatmap_to_coord(hm[i, :, :, eval_joints], bbox, hm_shape=hm_size,
                                                  norm_type=None)

        pose_coords.append(pose_coord)
        pose_scores.append(pose_score)
    
    preds_img = np.array(pose_coords)
    preds_scores = np.array(pose_scores)

    #preds_img = pose_coords  #torch.cat(pose_coords)
    #preds_scores = pose_scores  #torch.cat(pose_scores)
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
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
