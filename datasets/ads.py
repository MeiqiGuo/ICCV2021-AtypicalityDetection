# coding=utf-8
# modify based on https://github.com/airsplay/lxmert

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import logging


class AdsDatasetWithMask(Dataset):
    """Ads dataset with mask regions."""

    def __init__(self, phase: str, img_folder: str, atypical_annotation_path: str, feat_folder: str):
        super(AdsDatasetWithMask, self).__init__()
        with open(os.path.join(atypical_annotation_path, 'annotations.json'), "rb") as f:
            atypical_annotations = json.load(f)
        self.held_out_images = {x for x in atypical_annotations}
        self.img_folder = img_folder
        self.feat_folder = feat_folder

        # data in format of list
        self.data = []
        if phase == "train":
            sub_folder_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        elif phase == "val":
            sub_folder_list = ["9"]
        elif phase == "test":
            sub_folder_list = ["10"]
        else:
            logging.error("Wrong phase {0}".format(phase))

        for sub_folder in sub_folder_list:
            for image_name in os.listdir(os.path.join(img_folder, sub_folder)):
                image_id = "/".join([sub_folder, image_name])
                if image_id not in self.held_out_images:
                    self.data.append(image_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return:
        """
        img_name = os.path.join(self.img_folder, self.data[idx])
        image = cv2.imread(img_name)
        img_h, img_w, _ = image.shape

        # Loading bottom-up-attention features
        feat = np.load(os.path.join(self.feat_folder, self.data[idx].split(".")[0] + ".npy"), allow_pickle=True)
        features = feat.item()["features"]
        boxes = feat.item()["boxes"]
        assert features.shape[0] == boxes.shape[0], "Shape error"

        # Normalize the boxes (to 0 ~ 1)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        return features, boxes


class AtypicalDatasetWithMask(Dataset):
    """Ads atypical test dataset with mask regions."""

    def __init__(self, img_folder: str, atypical_annotation_path: str, feat_folder: str):
        super(AtypicalDatasetWithMask, self).__init__()
        with open(atypical_annotation_path, "rb") as f:
            atypical_annotations = json.load(f)
        self.img_folder = img_folder
        self.feat_folder = feat_folder

        self.data = []
        for img_id in atypical_annotations:
            datum = {}
            datum["img_id"] = img_id
            datum["label"] = self.fix_label(atypical_annotations[img_id])
            self.data.append(datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """

        :param idx:
        :return: tensor in shape (3, image_size, image_size)
        """
        datum = self.data[idx]
        img_name = os.path.join(self.img_folder, datum["img_id"])
        image = cv2.imread(img_name)
        img_h, img_w, _ = image.shape

        # Loading bottom-up-attention features
        feat = np.load(os.path.join(self.feat_folder, datum["img_id"].split(".")[0] + ".npy"), allow_pickle=True)
        features = feat.item()["features"]
        boxes = feat.item()["boxes"]
        assert features.shape[0] == boxes.shape[0], "Shape error"

        # Normalize the boxes (to 0 ~ 1)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        return datum["img_id"], features, boxes, datum["label"]

    def fix_label(self, label):
        if label == "Regular_Object":
            return 0
        elif label in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return 1
        else:
            assert False, "Error: unrecognized label {}".format(label)