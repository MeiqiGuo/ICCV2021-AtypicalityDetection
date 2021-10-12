"""
Various positional encodings for the transformer.
"""

import numpy as np
import math
import logging


def compute_center(box):
    """

    :param box: (box[0], box[1]) is the left low conner coordinate; (box[2], box[3]) is the right top conner coordinate.
    :return:
    """
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    return x, y


def compute_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def bb_intersection_over_union(boxA, boxB):
    assert boxA[0] < boxA[2]
    assert boxA[1] < boxA[3]
    assert boxB[0] < boxB[2]
    assert boxB[1] < boxB[3]

    # determinie the (x, y) - coordinates of the intersection rectangle
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    # area of the intersection rectangle
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    bb2_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(bb1_area + bb2_area - inter_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def relative_cartesian_coordinate(query_box, key_box, return_iou=False):
    relative_box = key_box - query_box
    if return_iou:
        IoU = bb_intersection_over_union(query_box, key_box)
        return np.append(relative_box, IoU)
    else:
        return relative_box


def relative_polar_coordinate(query_box, key_box):
    query_center = compute_center(query_box)
    key_center = compute_center(key_box)
    distance = np.sqrt((key_center[0] - query_center[0])**2 + (key_center[1] - query_center[1])**2)
    radian = math.atan2(key_center[1] - query_center[1], key_center[0] - query_center[0])   #[-3.14, 3.14]
    key_area = compute_area(key_box)
    query_area = compute_area(query_box)
    return np.array([distance, radian, key_area, query_area])


def compute_relative_position(boxes, pos_dim=4, is_cartesian=False, is_polar=False, iou=False):
    """

    :param boxes: (bsz, seq_len, 4)
    :param is_cartesian:
    :param is_polar:
    :return: pos: np.arrange(bsz, seq_len, seq_len, pos_dim)
             pos_dim: int
    """
    bsz = len(boxes)
    box_num = boxes.shape[1]
    pos = np.empty([bsz, box_num, box_num, pos_dim])
    for i, box in enumerate(boxes):
        for j, query_box in enumerate(box):
            for k, key_box in enumerate(box):
                if is_cartesian:
                    relative_positions = relative_cartesian_coordinate(query_box, key_box, iou)
                elif is_polar:
                    relative_positions = relative_polar_coordinate(query_box, key_box)
                    pass
                pos[i][j][k] = relative_positions
    return pos, pos_dim