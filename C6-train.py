#!/usr/bin/env python3
""" Script for training a face swapping model using PyTorch """

from glob import glob
import os
import random
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

from lib.models import OriginalEncoder, OriginalDecoder


def calcloss(original, out):
    """ Calculate the mean standard error between an original image and a predicted image. """
    return torch.abs(out-original).mean()


def main(opt):
    """ Train a Deepfake model from two folders of extracted face images.

        Example CLI:
        ------------
        python c6-train.py "C:/media_files/face1" "C:/media_files/face2"
    """
    device = "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"

    encoder = OriginalEncoder()
    decodera = OriginalDecoder()
    decoderb = OriginalDecoder()

    os.makedirs(opt.export_path, exist_ok=True)

    if os.path.exists(os.path.join(opt.export_path, "encoder.pth")):
        encoder.load_state_dict(
            torch.load(
                os.path.join(opt.export_path, "encoder.pth")).state_dict(),
                map_location=torch.device(device))
        decodera.load_state_dict(
            torch.load(
                os.path.join(opt.export_path, "decodera.pth")).state_dict(),
                map_location=torch.device(device))
        decoderb.load_state_dict(
            torch.load(
                os.path.join(opt.export_path, "decoderb.pth")).state_dict(),
                map_location=torch.device(device))

    imagesa = glob(os.path.join(opt.patha, "face_aligned_*.png"))
    imagesb = glob(os.path.join(opt.pathb, "face_aligned_*.png"))

    img_tensora = torch.zeros([opt.batchsize, 3, 64, 64])
    img_tensorb = torch.zeros([opt.batchsize, 3, 64, 64])
    mask_tensora = torch.zeros([opt.batchsize, 1, 64, 64])
    mask_tensorb = torch.zeros([opt.batchsize, 1, 64, 64])

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.learning_rate/2)
    decodera_optimizer = torch.optim.Adam(decodera.parameters(), lr=opt.learning_rate)
    decoderb_optimizer = torch.optim.Adam(decoderb.parameters(), lr=opt.learning_rate)
    loss_function = MSELoss()

    if device == "cuda":
        encoder = encoder.cuda()
        decodera = decodera.cuda()
        decoderb = decoderb.cuda()
        img_tensora = img_tensora.cuda()
        img_tensorb = img_tensorb.cuda()
        mask_tensora = mask_tensora.cuda()
        mask_tensorb = mask_tensorb.cuda()

    pbar = tqdm(range(opt.iterations))

    for iteration in pbar:
        images = random.sample(imagesa, opt.batchsize)
        for imgnum, imagefile in enumerate(images):
            image = cv2.imread(imagefile)
            image = cv2.resize(image, (64, 64))
            mask = cv2.imread(imagefile.replace("aligned", "mask"), 0)
            mask = cv2.resize(mask, (64, 64))
            if np.random.rand() > .5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            img_tensor = torch.tensor(image[..., ::-1]/255).permute(2, 0, 1)
            mask_tensor = torch.where(torch.tensor(mask) > 200, 1, 0)
            if device == "cuda":
                img_tensora[imgnum] = img_tensor.cuda()
                mask_tensora[imgnum] = mask_tensor.cuda()
        images_b = random.sample(imagesb, opt.batchsize)
        for imgnum, imagefile in enumerate(images_b):
            image = cv2.imread(imagefile)
            image = cv2.resize(image, (64, 64))
            mask = cv2.imread(imagefile.replace("aligned", "mask"), 0)
            mask = cv2.resize(mask, (64, 64))
            if np.random.rand() > .5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            img_tensor = torch.tensor(image[..., ::-1]/255).permute(2, 0, 1)
            mask_tensor = torch.tensor(mask/255).unsqueeze(0)
            if device == "cuda":
                img_tensorb[imgnum] = img_tensor.cuda()
                mask_tensorb[imgnum] = mask_tensor.cuda()

        encoder_optimizer.zero_grad()
        decodera_optimizer.zero_grad()

        outa = decodera(encoder(img_tensora))

        lossa = loss_function(outa * mask_tensora, img_tensora * mask_tensora)
        lossa.backward()
        encoder_optimizer.step()
        decodera_optimizer.step()

        encoder_optimizer.zero_grad()
        decoderb_optimizer.zero_grad()

        outb = decoderb(encoder(img_tensorb))

        lossb = loss_function(outb * mask_tensorb, img_tensorb * mask_tensorb)
        lossb.backward()
        encoder_optimizer.step()
        decoderb_optimizer.step()

        pbar.set_description(f"A: {lossa.detach().cpu().numpy():.6f} "
                             f"B: {lossb.detach().cpu().numpy():.6f}")

        if iteration % opt.save_freq == 0:
            with torch.no_grad():
                outa = decodera(encoder(img_tensora[:1]))
                outb = decoderb(encoder(img_tensorb[:1]))
                swapa = decoderb(encoder(img_tensora[:1]))[0]
                swapb = decodera(encoder(img_tensorb[:1]))[0]
                example = np.concatenate(
                    [img_tensora[0].permute(1, 2, 0).float().detach().cpu().numpy(),
                     outa[0].permute(1, 2, 0).float().detach().cpu().numpy(),
                     swapa.permute(1, 2, 0).float().detach().cpu().numpy(),
                     img_tensorb[0].permute(1, 2, 0).float().detach().cpu().numpy(),
                     outb[0].permute(1, 2, 0).float().detach().cpu().numpy(),
                     swapb.permute(1, 2, 0).float().detach().cpu().numpy()], axis=1)
                cv2.imwrite(os.path.join(opt.export_path, f"preview_{iteration}.png"),
                            example[..., ::-1] * 255)
                torch.save(encoder, os.path.join(opt.export_path, "encoder.pth"))
                torch.save(decodera, os.path.join(opt.export_path, "decodera.pth"))
                torch.save(decoderb, os.path.join(opt.export_path, "decoderb.pth"))

    torch.save(encoder, os.path.join(opt.export_path, "encoder.pth"))
    torch.save(decodera, os.path.join(opt.export_path, "decodera.pth"))
    torch.save(decoderb, os.path.join(opt.export_path, "decoderb.pth"))


if __name__ == "__main__":
    # Train a Deepfake model from two folders of extracted face images.
    #    Example CLI:
    #    ------------
    #    python c6-train.py "C:/media_files/face1" "C:/media_files/face2"
    parser = ArgumentParser()
    parser.add_argument("patha",
                        help="folder of images of face a")
    parser.add_argument("pathb",
                        help="folder of images of face b")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use Mixed Precision")
    parser.add_argument("--batchsize",
                        type=int,
                        default=16,
                        help="Number of images to include in a batch")
    parser.add_argument("--iterations",
                        type=int,
                        default=100000,
                        help="Number of iterations to process before stopping")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=.000001,
                        help="Number of images to include in a batch")
    parser.add_argument("--save_freq",
                        type=int,
                        default=1000,
                        help="Number of iterations to save between")
    parser.add_argument("--export_path",
                        default="model/",
                        help="folder to place models")

    opt = parser.parse_args()  # Parse all options to create an opt file

    main(opt)
