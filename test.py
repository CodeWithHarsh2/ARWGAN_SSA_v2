import torch.nn
import argparse
import os
import math
import utils
from model.ARWGAN import *
import numpy as np
import torch
from noise_argparser import NoiseArgParser
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF

import pandas as pd



def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def yuv_psnr(img):
    imgy = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2:, :, :]
    imgu = -0.14713 * img[:, 0, :, :] + (-0.28886) * img[:, 1, :, :] + 0.436 * img[:, 2:, :, :]
    imgv = 0.615 * img[:, 0, :, :] + -0.51499 * img[:, 1, :, :] + (-0.10001) * img[:, 2:, :, :]
    return imgy, imgu, imgv

def bit_accuracy(decoded, gt):
    decoded = decoded.round().clip(0,1)
    return np.mean(decoded == gt)


def ber(decoded, gt):
    decoded = decoded.round().clip(0,1)
    return np.mean(np.abs(decoded-gt))


def mse(img1,img2):
    return np.mean((img1-img2)**2)


def psnr(img1,img2):
    m = mse(img1,img2)

    if m==0:
        return 100

    return 20*np.log10(2.0/np.sqrt(m))


def ncc(decoded,gt):

    decoded=decoded.flatten()
    gt=gt.flatten()

    decoded=decoded-decoded.mean()
    gt=gt-gt.mean()

    return np.sum(decoded*gt)/(np.sqrt(np.sum(decoded**2))*np.sqrt(np.sum(gt**2))+1e-8)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source_images', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument("--noise", '-n', nargs="*", action=NoiseArgParser)
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()
    train_options, net_config, noise_config = utils.load_options(args.options_file)
    noise_config = args.noise
    noiser = Noiser(noise_config, device)

    checkpoint = torch.load(
    args.checkpoint_file,
    map_location=torch.device('cpu'))
    hidden_net = ARWGAN(net_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    source_images = os.listdir(args.source_images)

    results=[]

    for source_image in source_images:

        image_pil=Image.open(
            os.path.join(args.source_images,source_image)
        ).convert("RGB")

        image_pil=image_pil.resize(
            (net_config.W,net_config.H)
        )

        image_tensor=TF.to_tensor(image_pil).to(device)

        image_tensor=image_tensor*2-1

        image_tensor=image_tensor.unsqueeze(0)

        message=torch.Tensor(
            np.random.choice(
                [0,1],
                (1,net_config.message_length)
            )
        ).to(device)

        losses,(encoded,noised,decoded)=hidden_net.validate_on_batch(
            [image_tensor,message]
        )

        decoded_np=decoded.detach().cpu().numpy()

        gt=message.cpu().numpy()

        enc=encoded.detach().cpu().numpy()

        cover=image_tensor.cpu().numpy()

        results.append({

            "Image":source_image,

            "BER":ber(decoded_np,gt),

            "Accuracy":bit_accuracy(decoded_np,gt),

            "PSNR":psnr(enc,cover),

            "SSIM":losses["encoded_ssim   "],

            "NCC":ncc(decoded_np,gt),

            "MSE":mse(enc,cover)

        })

    #utils.save_images(image_tensor.cpu(), encoded_images.cpu(), 'test', '.', resize_to=(128, 128))

    df=pd.DataFrame(results)

    print(df)

    print()

    print(df.mean(numeric_only=True))

    df.to_csv("evaluation_results.csv",index=False)

    print("Saved evaluation_results.csv")


if __name__ == '__main__':
    main()
