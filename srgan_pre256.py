import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TFメッセージ非表示

import scipy
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

        input_shape = (256, 256, 3)

        # Pre stage(Down Sampling)
        self.pre = [
            kl.Conv2D(64, kernel_size=9, strides=1,
                    padding="same", input_shape=input_shape),
            kl.Activation(tf.nn.relu)
        ]

        # Residual Block
        self.res = [
            [
                Res_block(64) for _ in range(6)
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
                Pixel_shuffer(128) for _ in range(2)
            ],
            kl.Conv2D(3, kernel_size=6, strides=4, padding="same", activation="tanh")
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

        input_shape = (256, 256, 64)

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

        input_shape = (256, 256, 3)
        
        self.conv1 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same")
        self.bn2 = kl.BatchNormalization()

        self.add = kl.Add()

    # forward proc
    def call(self, x):
       
        d1 = self.av1(self.bn1(self.conv1(x)))
        d2 = self.bn2(self.conv2(d1))

        return self.add([x, d2])

def main():
    
    model = Generator()
    model.build((None, 256, 256, 3))
    model.load_weights('super.h5')

    img = cv2.imread("super_img/im_14.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_s = img

    img_low = cv2.resize(img, (32, 32))
    img_low = cv2.resize(img_low, (256, 256))
    img_low_s = img_low

    img = tf.convert_to_tensor(img, np.float32) 
    img_low = tf.convert_to_tensor(img_low, np.float32) 
    
    img_low = (img_low - 127.5) / 127.5
    img_low = img_low[np.newaxis, :, :, :]

    re = model.predict(img_low)
    
    re = np.reshape(re, (256, 256, 3))
    re = re * 127.5 + 127.5

    img_s = Image.fromarray(np.uint8(img_s))
    img_s.show()
    img_s.save("result/High-resolution Image256.bmp")

    img_low_s = Image.fromarray(np.uint8(img_low_s))
    img_low_s.show()
    img_low_s.save("result/Low-resolution Image256.bmp")

    re = Image.fromarray(np.uint8(re))
    re.show()
    re.save("result/Super-resolution Image256.bmp")
    

    
if __name__ == "__main__":
    main()