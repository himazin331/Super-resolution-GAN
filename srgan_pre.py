import tensorflow as tf
import tensorflow.keras.layers as kl

import cv2
from PIL import Image
import numpy as np

import argparse as arg
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Super-resolution Image Generator
class Generator(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        input_shape_ps = (input_shape[0], input_shape[1], 64)

        # Pre stage(Down Sampling)
        self.pre = [
            kl.Conv2D(64, kernel_size=9, strides=1,
                        padding="same", input_shape=input_shape),
            kl.Activation(tf.nn.relu)
        ]

        # Residual Block
        self.res = [
            [
                Res_block(64, input_shape) for _ in range(7)
            ]
        ]

        # Middle stage
        self.middle = [
            kl.Conv2D(64, kernel_size=3, strides=1, padding="same"),
            kl.BatchNormalization()
        ]

        # Pixel Shuffle(Up Sampling)
        self.ps = [
            [
                Pixel_shuffler(128, input_shape_ps) for _ in range(2)
            ],
            kl.Conv2D(3, kernel_size=9, strides=4, padding="same", activation="tanh")
        ]

    def call(self, x):
        # Pre stage
        pre = x
        for layer in self.pre:
            pre = layer(pre)

        # Residual Block
        res = pre
        for layer in self.res:
            for _layer in layer:
                res = _layer(res)
        
        # Middle stage
        middle = res
        for layer in self.middle:
            middle = layer(middle)
        middle += pre

        # Pixel Shuffle
        out = middle
        for layer in self.ps:
            if isinstance(layer, list):
                for _layer in layer:
                    out = _layer(out)
            else:
                out = layer(out)

        return out


# Pixel Shuffle
class Pixel_shuffler(tf.keras.Model):
    def __init__(self, out_ch, input_shape):
        super().__init__()

        self.conv = kl.Conv2D(out_ch, kernel_size=3, strides=1,
                                padding="same", input_shape=input_shape)
        self.act = kl.Activation(tf.nn.relu)

    # forward proc
    def call(self, x):
        d1 = self.conv(x)
        d2 = self.act(tf.nn.depth_to_space(d1, 2))

        return d2


# Residual Block
class Res_block(tf.keras.Model):
    def __init__(self, ch, input_shape):
        super().__init__()

        self.conv1 = kl.Conv2D(ch, kernel_size=3, strides=1,
                                padding="same", input_shape=input_shape)
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(ch, kernel_size=3, strides=1,
                                padding="same")
        self.bn2 = kl.BatchNormalization()

        self.add = kl.Add()

    def call(self, x):
        d1 = self.av1(self.bn1(self.conv1(x)))
        d2 = self.bn2(self.conv2(d1))

        return self.add([x, d2])


def main():
    # Command line option
    parser = arg.ArgumentParser(description='Super-resolution GAN prediction')
    parser.add_argument('--param', '-p', type=str, default=None,
                        help='Specify learned parameters (If not specified, an error)')
    parser.add_argument('--data_img', '-d', type=str, default=None,
                        help='Specify an image file (If not specified, an error)')
    parser.add_argument('--out', '-o', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "result"),
                        help='Specify the destination (default: ./result)')
    parser.add_argument('--he', '-he', type=int, default=128,
                        help='Resize height (default: 128)')
    parser.add_argument('--wi', '-wi', type=int, default=128,
                        help='Resize width (default: 128)')
    parser.add_argument('--mag', '-m', type=int, default=2,
                        help='Magnification (default: 2)')
    args = parser.parse_args()

    # Parameter-File not specified. -> Exception
    if args.param is None:
        print("\nException: Trained Parameter-File not specified.\n")
        sys.exit()
    # An Parameter-File that does not exist was specified. -> Exception
    if os.path.exists(args.param) is False:
        print("\nException: Trained Parameter-File {} is not found.\n".format(args.param))
        sys.exit()
    # Image not specified. -> Exception
    if args.data_img is None:
        print("\nException: Image not specified.\n")
        sys.exit()
    # An image that does not exist was specified. -> Exception
    if os.path.exists(args.data_img) is False:
        print("\nException: Image {} is not found.\n".format(args.data_img))
        sys.exit()
    # When 0 is entered for either width/height or Reduction ratio. -> Exception
    if args.he == 0 or args.wi == 0 or args.mag == 0:
        print("\nException: Invalid value has been entered.\n")
        sys.exit()

    # Setting info
    print("=== Setting information ===")
    print("# Trained Prameter-File: {}".format(os.path.abspath(args.param)))
    print("# Image: {}".format(args.data_img))
    print("# Output folder: {}".format(args.out))
    print("")
    print("# Height: {}".format(args.he))
    print("# Width: {}".format(args.wi))
    print("# Magnification: {}".format(args.mag))
    print("===========================")

    # Create output folder (If the folder exists, it will not be created.)
    os.makedirs(args.out, exist_ok=True)

    # Network Setup
    model = Generator(input_shape=(args.he, args.wi, 3))
    model.build((None, args.he, args.wi, 3))
    model.load_weights(args.param)

    # High-resolutin Image
    img = cv2.imread(args.data_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hr_img = cv2.resize(img, (args.he, args.wi))

    # Low-resolution Image
    lr_img = cv2.resize(hr_img, (int(args.he / args.mag), int(args.wi / args.mag)))
    lr_img = cv2.resize(lr_img, (args.he, args.wi))
    lr_img_s = lr_img

    # Image processing
    lr_img = tf.convert_to_tensor(lr_img, np.float32)
    lr_img = tf.convert_to_tensor(lr_img, np.float32)
    lr_img = (lr_img - 127.5) / 127.5
    lr_img = lr_img[np.newaxis, :, :, :]

    # Super-resolution
    re = model.predict(lr_img)
    
    # Super-resolution Image processing
    re = np.reshape(re, (args.he, args.wi, 3))
    re = re * 127.5 + 127.5
    re = np.clip(re, 0.0, 255.0)

    # Low-resolution Image output
    lr_img = Image.fromarray(np.uint8(lr_img_s))
    lr_img.show()
    lr_img.save(os.path.join(args.out, "Low-resolution Image(SRGAN).bmp"))

    # Super-resolution Image output
    sr_img = Image.fromarray(np.uint8(re))
    sr_img.show()
    sr_img.save(os.path.join(args.out, "Super-resolution Image(SRGAN).bmp"))

    # High-resolution Image output
    hr_img = Image.fromarray(np.uint8(hr_img))
    hr_img.show()
    hr_img.save(os.path.join(args.out, "High-resolution Image(SRGAN).bmp"))


if __name__ == "__main__":
    main()
