import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TFメッセージ非表示


import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.preprocessing.image as kp
from tensorflow.python.keras import backend as K
from IPython.display import display_png
import cv2
from PIL import Image
import numpy as np

# Super-resolution Image Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (128, 128, 3)

        # Pre stage(Down Sampling)
        self.pre = [
            kl.Conv2D(64, kernel_size=9, strides=1,
                    padding="same", input_shape=input_shape),
            kl.Activation(tf.nn.relu)
        ]

        # Residual Block
        self.res = [
            [
                Res_block(64) for _ in range(7)
            ]
        ]

        # Middle stage
        self.middle = [
            kl.Conv2D(64, kernel_size=3, strides=1, padding="same"),
            kl.BatchNormalization()
        ]

        # Pixel Shuffle(Up Sampling)
        self.ps =[
            [
                Pixel_shuffer(256) for _ in range(2)
            ],
            kl.Conv2D(3, kernel_size=9, strides=4, padding="same", activation="tanh")
        ]

    # forward proc
    def call(self, x):

        # Pre stage
        pre = x
        for layer in self.pre:
            pre = layer(pre)

        # Residual Block
        res = pre
        for layer in self.res:
            for l in layer:
                res = l(res)
        
        # Middle stage
        middle = res
        for layer in self.middle:
            middle = layer(middle)
        middle += pre

        # Pixel Shuffle
        out = middle
        for layer in self.ps:
            if isinstance(layer, list):
                for l in layer:
                    out = l(out)
            else:
                out = layer(out)

        return out

# Pixel Shuffle
class Pixel_shuffer(tf.keras.Model):
    def __init__(self, out_ch):
        super().__init__()

        input_shape = (128, 128, 256)

        self.up = kl.UpSampling2D(size=2)
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
    def __init__(self, ch):
        super().__init__()

        input_shape = (128, 128, 3)
        
        self.conv1 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same")
        self.bn2 = kl.BatchNormalization()

    # forward proc
    def call(self, x):

        d1 = self.av1(self.bn1(self.conv1(x)))
        d2 = self.bn2(self.conv2(d1))

        return x + d2

def main():
    
    # Network Setup
    model = Generator()
    model.build((None, 128, 128, 3))
    model.load_weights('super.h5')

    # High-resolutin Image
    img = cv2.imread("super_img/test1.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img_s = img

    # Low-resolution Image
    img_low = cv2.resize(img, (64, 64))
    img_low = cv2.resize(img_low, (128, 128))
    img_low_s = img_low

    # Image processing
    img = tf.convert_to_tensor(img, np.float32) 
    img_low = tf.convert_to_tensor(img_low, np.float32) 
    img_low = (img_low - 127.5) / 127.5
    img_low = img_low[np.newaxis, :, :, :]

    # Super-resolution
    re = model.predict(img_low)
    
    # Super-resolution Image processing
    re = np.reshape(re, (128, 128, 3))
    re = re * 127.5 + 127.5

    # High-resolution Image output
    img_s = Image.fromarray(np.uint8(img_s))
    img_s.show()
    img_s.save("result/High-resolution Image.bmp")

    # Low-resolution Image output
    img_low_s = Image.fromarray(np.uint8(img_low_s))
    img_low_s.show()
    img_low_s.save("result/Low-resolution Image.bmp")

    # Super-resolution Image output
    re = Image.fromarray(np.uint8(re))
    re.show()
    re.save("result/Super-resolution Image.bmp")
    
if __name__ == "__main__":
    main()