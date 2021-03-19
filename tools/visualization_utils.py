import math
import cv2
import numpy as np


def visualize_poses(img, poses):
    kp_num = 17
    l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    colors = [(84, 46, 195), (240, 143, 189), (139, 213, 241), (240, 104, 168),
                    (109, 46, 195), (46, 134, 195), (104, 172, 242), (252, 48, 125), (199, 79, 136),
                    (46, 183, 195), (253, 103, 174),
                    (28, 57, 220), (240, 58, 145), (53, 46, 195), (200, 140, 168)]
    num_humans = len(poses)
    human_ratio = num_humans // len(colors)
    colors = colors * (human_ratio + 1)
    height, width = img.shape[:2]
    bg = img.copy()

    for i, human in enumerate(poses):
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        line_width = 2
        transparency = 0.55
        if kp_scores[0] > 0.4 and kp_scores[-1] > 0.4:
            human_height = np.abs(kp_preds[-1, 1] - kp_preds[0, 1])
            height_ratio = human_height / height
            transparency = np.clip(1 - height_ratio, 0.55, 1)
            line_width = int(np.clip(2 * height_ratio, 1, 20))
        kp_preds = np.vstack((kp_preds, (kp_preds[5, :] + kp_preds[6, :]) / 2))
        kp_scores = np.vstack((kp_scores, (kp_scores[5, :] + kp_scores[6, :]) / 2))
        vis_thres = 0.4
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            cv2.circle(bg, (int(cor_x), int(cor_y)), line_width, colors[i], -1)

        for j, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(bg, start_xy, end_xy, colors[i], line_width)
    transparency = 0.55
    img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    return img





