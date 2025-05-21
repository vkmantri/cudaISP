#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path


def simple_white_balance(image):
    # Calculate the mean of each color channel
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])

    # Calculate scaling factors for each channel
    scale_b = avg_g / avg_b
    scale_r = avg_g / avg_r

    # Apply the scaling factors to balance the image
    balanced_image = cv2.merge(
        [
            np.clip(image[:, :, 0] * scale_b, 0, 255).astype(np.uint8),
            np.clip(image[:, :, 1], 0, 255).astype(np.uint8),
            np.clip(image[:, :, 2] * scale_r, 0, 255).astype(np.uint8),
        ]
    )

    return balanced_image


def choose_opencv_bayer_pattern(pattern):
    if pattern == "GRBG":
        return cv2.COLOR_BAYER_GR2BGR
    elif pattern == "RGGB":
        return cv2.COLOR_BAYER_RG2BGR
    elif pattern == "GBRG":
        return cv2.COLOR_BAYER_GB2BGR
    elif pattern == "BGGR":
        return cv2.COLOR_BAYER_BG2BGR
    else:
        raise ValueError("Invalid Bayer pattern")


def main(raw_image, width, height, gamma=0.3, output="", bayer_pattern="GRBG"):
    bayer_im = np.fromfile(Path(raw_image), "uint16").reshape(height, width)
    # This is useful for cropping the image when you use the preferred_stride v4l2 arg
    # if width > 1944:
    #     bayer_im = bayer_im[:, :1944]
    print(f"Input size: ({height}, {width}), output size: {bayer_im.shape}")
    color_conversion = choose_opencv_bayer_pattern(bayer_pattern)

    # Per the Nvidia Orin TRM, RAW12 pixel data is stored left justified in a 16-bit word
    # when T_R16 is used (see vi5_formats.h in the kernel).
    # 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
    # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    # |11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|11|10| 9| 8|
    # +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    # So we right shift the data to get rid of the duplicate bits
    bayer_im = np.right_shift(bayer_im, 4)

    # The result is BGR format with 16 bits per pixel and 12 bits range [0, 2^12-1].
    bgr = cv2.cvtColor(bayer_im, color_conversion)
    gamma = 0.30
    norm_gain = 1.0 / np.max(bgr)
    bgr = pow(bgr * norm_gain, gamma)
    wb = simple_white_balance((bgr * 255).astype("uint8"))
    if not output:
        output = os.path.split(raw_image)[-1] + ".png"
    plt.imsave(Path(output), wb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RAW12 GRBG image to PNG")
    parser.add_argument("raw_image", help="Input RAW12 image file")
    parser.add_argument("-W", "--width", type=int, default=1944, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=1204, help="Image height")
    parser.add_argument(
        "-g", "--gamma", type=float, default=0.30, help="Gamma correction value"
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Output PNG file")
    parser.add_argument(
        "-p", "--bayer-pattern", type=str, default="GRBG", help="Bayer pattern"
    )
    args = parser.parse_args()
    main(
        args.raw_image,
        args.width,
        args.height,
        args.gamma,
        args.output,
        args.bayer_pattern,
    )
