import os
import cv2
import torch
import dataset
import util
import argparse
import numpy as np
import magic_wand_model
from pprint import pprint
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='magic wand inference script')
    parser.add_argument('--data_path', '-d', type=str, default='data/test', help='path to dataset')
    parser.add_argument('--model_path', '-m', type=str, default='model', help='model path')
    args = parser.parse_args()
    return args


def test(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = magic_wand_model.magic_wand_model(num_classes=len(util.classes)).to(device)

    if os.path.exists(os.path.join(args.model_path, "best.pt")):
        checkpoint = torch.load(os.path.join(args.model_path, "best.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])

    image = cv2.imread(args.data_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('image', np.array(image))
    # cv2.waitKey(0)

    image = (255 - image) / 255.0
    image = cv2.resize(image, (28, 28))
    image = torch.tensor(image, dtype=torch.float32).to(device)

    model.eval()
    output = model(image.unsqueeze(0).unsqueeze(0))
    print(output, util.classes[torch.argmax(output)])


if __name__ == '__main__':
    args = get_args()
    test(args)
