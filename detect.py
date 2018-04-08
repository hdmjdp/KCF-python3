# -*- coding: utf-8 -*-
# file: face_solver.py
# author: hudameng
# time: 04/03/2018 11:16 PM
# Copyright 2018 hudameng. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# from modules.base.solver import Solver
import cv2
import mxnet as mx
import numpy as np
from detector_model.mtcnn_detector import MtcnnDetector
import dlib
import os

def mtcnn_detect(detector, img_bgr, image_size=160):
    results = detector.detect_face(img_bgr)
    bboxs = []

    draw = img_bgr.copy()

    if results is not None:
        total_boxes = results[0]
        points = results[1]

        # extract aligned face chips
        chips = detector.extract_image_chips(
            img_bgr, points, image_size, 0.37)

        faces = np.empty((len(chips), image_size, image_size, 3))
        blurs = []
        for i, chip in enumerate(chips):
            blur = cv2.Laplacian(chip, cv2.CV_64F).var()
            blurs.append(blur)
            faces[i, :, :, :] = chip

        for i, b in enumerate(total_boxes):
            bb = np.zeros(6)
            bb[0], bb[1], bb[2], bb[3] = int(
                b[0]), int(b[1]), int(b[2]), int(b[3])
            bbox = (bb[0], bb[1], bb[2], bb[3], b[4], blurs[i])
            bboxs.append(bbox)
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(
                b[2]), int(b[3])), (255, 255, 255))

        # for p in points:
        #     for i in range(5):
        #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
        return draw, faces, bboxs
    else:
        return img_bgr, None, None


if __name__ == "__main__":
    detector_model_dir = os.path.join('/Volumes/Transcend/jintian/KCF-python3/detector_model/model')
    detector = MtcnnDetector(model_folder=detector_model_dir, minsize=40, threshold=[0.8, 0.8, 0.9], ctx=mx.cpu(0),
                             num_worker=4,
                             accurate_landmark=False)
    frame = cv2.imread("penguin.jpg")
    img_bgr, faces, faceRect = mtcnn_detect(detector, frame)

    cv2.imshow('z', img_bgr)
    cv2.waitKey(0)
