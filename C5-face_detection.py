#!/usr/bin/env python3
""" Script for extracting faces from images using PyTorch """

import os

from argparse import ArgumentParser

import torch
import cv2
import json_tricks
import numpy as np
from tqdm import tqdm

from face_alignment.detection.sfd import FaceDetector
from face_alignment import FaceAlignment, LandmarksType

from skimage.transform._geometric import _umeyama as umeyama

from lib.bisenet import BiSeNet


def main(opt):
    """ Process images in a directory into aligned face images

       Example CLI:
       ------------
       python C5-face_detection.py "C:/media_files/"
    """
    if not os.path.exists(opt.export_path):
        os.mkdir(opt.export_path)

    device = "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"

    face_detector = FaceDetector(device=device, verbose=False)
    face_aligner = FaceAlignment(LandmarksType._2D, device=device, verbose=False)

    masker = BiSeNet(n_classes=19)
    if device == "cuda":
        masker.cuda()
    model_path = os.path.join(".", "binaries", "BiSeNet.pth")
    masker.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    masker.eval()
    desired_segments = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]

    # Get a list of all files in the specified path.  Only images should be in the folder
    alignment_data = {}
    list_of_images_in_dir = [file for file in os.listdir(opt.path)
                             if os.path.isfile(os.path.join(opt.path, file))]

    # Iterate over all the files found
    for file in tqdm(list_of_images_in_dir):
        filename, extension = os.path.splitext(file)
        image_bgr = cv2.imread(os.path.join(opt.path, file))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        adjustment = opt.max_detection_size / max(height, width)

        if adjustment < 1.0:
            resized_image = cv2.resize(image_rgb, None, fx=adjustment, fy=adjustment)
        else:
            resized_image = image_rgb
            adjustment = 1.0

        faces = face_detector.detect_from_image(resized_image)

        for idx, face in enumerate(faces):
            top, left, bottom, right = (face[0:4] / adjustment).astype("int")
            top = max(0, top)
            left = max(0, left)
            bottom = min(height, bottom)
            right = min(width, right)

            confidence = face[4]
            if confidence < opt.min_confidence:
                continue

            face_height = bottom - top
            face_width = right - left
            face_size = face_height * face_width
            if face_size / (height * width) < opt.min_size:
                continue

            detected_face = image_bgr[left:right, top:bottom]
            # cv2.imwrite(os.path.join(opt.export_path,
            #             f"face_bbox_{filename}_{idx}.png"),
            #             detected_face)

            landmarks = face_aligner.get_landmarks_from_image(
                image_rgb,
                detected_faces=[face[0:4] / adjustment])

            landmark_image = image_bgr.copy()
            landmark_image = cv2.rectangle(landmark_image,
                                           (top, left),
                                           (bottom, right),
                                           thickness=10,
                                           color=(0, 0, 0))

            right_eye = np.mean(landmarks[0][36:42], axis=0)
            left_eye = np.mean(landmarks[0][42:48], axis=0)
            nose_tip = landmarks[0][30]
            right_mouth = landmarks[0][48]
            left_mouth = landmarks[0][54]

            limited_landmarks = np.stack((right_eye, left_eye, nose_tip, right_mouth, left_mouth))

            colors = [[255, 0, 0],  # Blue
                      [0, 255, 0],  # Green
                      [0, 0, 255],  # Red
                      [255, 255, 0],  # Cyan
                      [0, 255, 255]]  # Yellow

            for count, landmark in enumerate(limited_landmarks):
                landmark_adjusted = landmark.astype(int)
                landmark_image = cv2.circle(landmark_image,
                                            tuple(landmark_adjusted),
                                            radius=10,
                                            thickness=-1,
                                            color=colors[count])

            # cv2.imwrite(os.path.join(opt.export_path,
            #             f"face_landmarks_{filename}_{idx}.png"),
            #             landmark_image)

            mean_face = np.array([[0.25, 0.22],
                                  [0.75, 0.22],
                                  [0.50, 0.51],
                                  [0.26, 0.78],
                                  [0.74, 0.78]])

            warp_matrix = umeyama(limited_landmarks,
                                  mean_face * (opt.size * .3) + (opt.size * .35),
                                  True)

            aligned_face = image_bgr.copy()

            aligned_face = cv2.warpAffine(aligned_face,
                                          warp_matrix[:2],
                                          (opt.size, opt.size))

            cv2.imwrite(os.path.join(opt.export_path,
                                     f"face_aligned_{filename}_{idx}.png"),
                        aligned_face)

            if file not in alignment_data.keys():
                alignment_data[file] = {"faces": list()}

            alignment_data[file]['faces'].append({"landmark": landmarks,
                                                  "warp_matrix": warp_matrix})

            mask_face = cv2.resize(aligned_face, (512, 512))
            mask_face = torch.tensor(mask_face, device=device).unsqueeze(0)
            mask_face = mask_face.permute(0, 3, 1, 2) / 255
            if device == "cuda":
                mask_face.cuda()
            segments = masker(mask_face)[0]

            segments = torch.softmax(segments, dim=1)
            segments = torch.nn.functional.interpolate(segments,
                                                       size=(256, 256),
                                                       mode="bicubic",
                                                       align_corners=False)
            mask = torch.where(torch.sum(segments[:, desired_segments, :, :], dim=1) > .7,
                               255,
                               0)[0]
            mask = mask.cpu().numpy()
            # cv2.imwrite(os.path.join(opt.export_path,
            #                          f"face_mask_{filename}_{idx}.png"),
            #             mask)

    with open(os.path.join(opt.export_path, "face_alignments.json"), "w",
              encoding="utf-8") as alignment_file:
        alignment_file.write(json_tricks.dumps(alignment_data, indent=4))


if __name__ == "__main__":
    # Process images in a directory into aligned face images
    #
    #   Example CLI:
    #   ------------
    #   python face_detection.py "C:/media_files/""

    parser = ArgumentParser()
    parser.add_argument("path",
                        help="folder of images to run detection on")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--size",
                        default=256,
                        type=int,
                        help="height and width to save the aligned face images")
    parser.add_argument("--max_detection_size",
                        default=1024,
                        type=int,
                        help="Maximum size of an image to run detection on.  If you get memory "
                             "errors, reduce this size")
    parser.add_argument("--jpg",
                        action="store_true",
                        help="use JPG instead of PNG for image saving (not recommended due to "
                             "compression artifacts in JPG images)")
    parser.add_argument("--min_size",
                        default=.01,
                        type=float,
                        help="Minimum relative size of the face in the image")
    parser.add_argument("--min_confidence",
                        default=.9,
                        type=float,
                        help="Minimum confidence for the face detection")
    parser.add_argument("--export-path",
                        default="$path/face_images",
                        help="folder to place face (replaces $path with the input path)")

    opt = parser.parse_args()
    opt.export_path = opt.export_path.replace("$path", opt.path)

    main(opt)
