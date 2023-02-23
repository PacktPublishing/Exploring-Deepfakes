#!/usr/bin/env python3
""" Script for extracting faces from images using PyTorch """

import os
from argparse import ArgumentParser
import json

import torch
import cv2
import numpy as np
from tqdm import tqdm

import face_alignment

from face_alignment.detection.sfd import FaceDetector
from face_alignment import FaceAlignment, LandmarksType

from lib.bisenet import BiSeNet

import json_tricks

from lib.models import OriginalEncoder, OriginalDecoder

def main(opt):
    if not os.path.exists(opt.export_path):
        os.mkdir(opt.export_path)

    device = "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"

    encoder = OriginalEncoder()
    decoder = OriginalDecoder()

    encoder.load_state_dict(
        torch.load(os.path.join(opt.model_path, "encoder.pth"),
                map_location=torch.device(device)).state_dict())
    if not opt.swap:
        decoder.load_state_dict(torch.load(os.path.join(opt.model_path, "decoderb.pth"),
                map_location=torch.device(device)).state_dict())
    else:
        decoder.load_state_dict(torch.load(os.path.join(opt.model_path, "decodera.pth"),
                map_location=torch.device(device)).state_dict())

    if device == "cuda":
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    with open(os.path.join(extract_path, "face_alignments.json"), "r", encoding="utf-8") as alignment_file:
        alignment_data = json_tricks.loads(alignment_file.read(), encoding="utf-8")
    alignment_keys = list(alignment_data.keys())

    # Get a list of all files in the specified path.  Only images should be in the folder
    list_of_images_in_dir = [file for file in os.listdir(opt.path)
                             if os.path.isfile(os.path.join(opt.path, file)) and file in alignment_keys]

    for file in tqdm(list_of_images_in_dir):
        filename, extension = os.path.splitext(file)
        image_bgr = cv2.imread(os.path.join(opt.path, file))
        width, height, channels = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        output_image = image_rgb

        for idx, face in enumerate(alignment_data[file]['faces']):
            aligned_face = cv2.warpAffine(image_rgb, face["warp_matrix"][:2], (256, 256))
            aligned_face_tensor = torch.tensor(aligned_face/255, dtype=torch.float32).permute(2, 0, 1)
            aligned_face_tensor_small = torch.nn.functional.interpolate(aligned_face_tensor.unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False)

            with torch.no_grad():
                output_face_tensor = decoder(encoder(aligned_face_tensor_small))

            output_face_tensor = torch.nn.functional.interpolate(output_face_tensor, size=(256,256))

            mask_img = cv2.imread(os.path.join(extract_path, f"face_mask_{filename}_{idx}.png"), 0)
            mask_tensor = torch.where(torch.tensor(mask_img) > 200, 1, 0)
            output_face_tensor = (aligned_face_tensor.cpu() * (1 - mask_tensor)) + (output_face_tensor.cpu() * mask_tensor)

            output_face = (output_face_tensor[0].permute(1,2,0).numpy()*255).astype(np.uint8)
            output_image = cv2.warpAffine(output_face, face["warp_matrix"][:2], (height, width), output_image, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)

        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(opt.export_path, f"{filename}.png"), output_image)

if __name__ == "__main__":
    """ Process images, replacing the face with another as trained

       Example CLI:
       ------------
       python C7-convert.py "C:/media_files/"
    """

    parser = ArgumentParser()
    parser.add_argument("path",
                        help="folder of images to convert")
    parser.add_argument("--model-path",
                        default="model/",
                        help="folder which has the trained model")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--swap",
                        action="store_true",
                        help="Convert to the first face trained instead of the second")
    parser.add_argument("--extract-path",
                        default="$path/face_images/",
                        help="path to the extract folder from C5-extract.py")
    parser.add_argument("--export-path",
                        default="$path/convert/",
                        help="folder to place face (replaces $path with the input path)")

    opt = parser.parse_args()
    opt.export_path = opt.export_path.replace("$path", opt.path)
    opt.extract_path = opt.extract_path.replace("$path", opt.path)

    main(opt)
