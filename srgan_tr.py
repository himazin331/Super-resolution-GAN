# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Super-resolution Image Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (128, 128, 3)

        # Pre stage(Down Sampling)
        self.pre = [
            kl.Conv2D(64, kernel_size=7, strides=1,
                    padding="same", input_shape=input_shape),
            kl.Activation(tf.nn.relu)
        ]

        # Residual Block
        self.res = [
            [
                Res_block(64) for _ in range(5)
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
                Pixel_shuffer(256, 2) for _ in range(2)
            ],
            kl.Conv2D(3, kernel_size=9, strides=4, padding="same")
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

# Discriminator 
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (128, 128, 3)


        self.conv1 = kl.Conv2D(64, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.act1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(64, kernel_size=3, strides=2,
                            padding="same")
        self.bn1 = kl.BatchNormalization()
        self.act2 = kl.LeakyReLU()

        self.conv3 = kl.Conv2D(128, kernel_size=3, strides=1,
                            padding="same")
        self.bn2 = kl.BatchNormalization()
        self.act3 = kl.LeakyReLU()

        self.conv4 = kl.Conv2D(128, kernel_size=3, strides=2,
                            padding="same")
        self.bn3 = kl.BatchNormalization()
        self.act4 = kl.LeakyReLU()

        self.conv5 = kl.Conv2D(256, kernel_size=3, strides=1,
                            padding="same")
        self.bn4 = kl.BatchNormalization()
        self.act5 = kl.LeakyReLU()

        self.conv6 = kl.Conv2D(256, kernel_size=3, strides=2,
                            padding="same")
        self.bn5 = kl.BatchNormalization()
        self.act6 = kl.LeakyReLU()

        self.conv7 = kl.Conv2D(512, kernel_size=3, strides=1,
                            padding="same")
        self.bn6 = kl.BatchNormalization()
        self.act7 = kl.LeakyReLU()

        self.conv8 = kl.Conv2D(512, kernel_size=3, strides=2,
                            padding="same")
        self.bn7 = kl.BatchNormalization()
        self.act8 = kl.LeakyReLU()

        self.flt = kl.Flatten()

        self.dens1 = kl.Dense(1024, activation=kl.LeakyReLU())
        self.dens2 = kl.Dense(1, activation="sigmoid")

    # forward proc
    def call(self, x):

        d1 = self.act1(self.conv1(x))
        d2 = self.act2(self.bn1(self.conv2(d1)))
        d3 = self.act3(self.bn2(self.conv3(d2)))
        d4 = self.act4(self.bn3(self.conv4(d3)))
        d5 = self.act5(self.bn4(self.conv5(d4)))
        d6 = self.act6(self.bn5(self.conv6(d5)))
        d7 = self.act7(self.bn6(self.conv7(d6)))
        d8 = self.act8(self.bn7(self.conv8(d7)))

        d9 = self.dens1(self.flt(d8))
        d10 = self.dens2(d9)

        return d10

# Pixel Shuffle
class Pixel_shuffer(tf.keras.Model):
    def __init__(self, out_ch, r):
        super().__init__()

        input_shape = (128, 128, 256)

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

# Train
class trainer():
    def __init__(self):

        lr_shape = (128, 128, 3)
        hr_shape = (128, 128, 3)

        # Content Loss Model setup
        input_tensor = tf.keras.Input(shape=hr_shape)
        vgg = VGG16(include_top=False, input_tensor=input_tensor)
        vgg.outputs = [vgg.layers[9].output]  # VGG16 block3_conv3 output  
        features = vgg(input_tensor) 

        # Content Loss Model
        self.cl_model = tf.keras.Model(input_tensor, features)
        self.cl_model.trainable = False
        self.cl_model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=['accuracy'])

        # Discriminator
        self.discriminator = Discriminator()
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'])

        # Generator
        self.generator = Generator()

        # Combined Model setup
        hr_input = tf.keras.Input(shape=hr_shape)
        lr_input = tf.keras.Input(shape=lr_shape)
        sr_output = self.generator(lr_input)
        sr_features = self.cl_model(sr_output)

        self.discriminator.trainable = False
        valid = self.discriminator(sr_output)

        # Combined
        self.combined = tf.keras.Model(inputs=[lr_input, hr_input], outputs=[valid, sr_features])
        self.combined.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanSquaredError()])
        print(lr_input)

    def train(self, lr_imgs, hr_imgs, out_path, batch_size, iteration):

        h_batch = int(batch_size / 2)

        real_lab = np.ones((h_batch, 1))  # High-resolution image label
        fake_lab = np.zeros((h_batch, 1)) # Super-resolution image label(Discriminator side)
        dummy_lab = np.ones((h_batch, 1)) # Super-resolution image label(Generator side)
        # train run
        for ite in range(iteration):

            # High-resolution image random pickups
            idx = np.random.randint(0, hr_imgs.shape[0], h_batch)
            hr_img = hr_imgs[idx]
            print(hr_img)
            # Low-resolution image random pickups
            idx = np.random.randint(0, lr_imgs.shape[0], h_batch)
            lr_img = lr_imgs[idx]

            # Discriminator enabled train
            self.discriminator.trainable = True

            # train by High-resolution image
            d_real_loss = self.discriminator.train_on_batch(hr_img, np.ones((h_batch, 1)))

            # train by Super-resolution image
            sr_img = self.generator.predict(lr_img)
            d_fake_loss = self.discriminator.train_on_batch(sr_img, fake_lab)
            
            # Discriminator average loss 
            d_loss = 0.5 * np.add(d_real_loss, d_fake_loss)

            # Discriminator disabled loss
            self.model.discriminator.trainable = False

            g_loss = self.model.train_on_batch(sr_img)

        
    def save_imgs(self, iteration):
        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % iteration)
        plt.close() 

    
        
def create_dataset(data_dir):

    lr_imgs = []
    hr_imgs = []

    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)

        _, ext = os.path.splitext(c)
        if ext.lower() != '.bmp':
            continue

        img = tf.io.read_file(d)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, (128, 128))

        img_low = tf.image.resize(img, (64, 64))
        img_low = tf.image.resize(img_low, (128, 128))

        img /= 255
        img_low /= 255

        lr_imgs.append(img_low)
        hr_imgs.append(img)

    lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32) 
    lr_imgs = lr_imgs.numpy()
    hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
    hr_imgs = hr_imgs.numpy()
    
    return lr_imgs, hr_imgs

def main():
    data_dir = "super_img"
    out_path = os.path.dirname(os.path.abspath(__file__))+'/super.h5'.replace('/', os.sep)
    batch_size = 32
    iteration = 3000

    lr_imgs, hr_imgs = create_dataset(data_dir)

    Trainer = trainer()
    Trainer.train(lr_imgs, hr_imgs, out_path=out_path, batch_size=batch_size, iteration=iteration)

if __name__ == '__main__':
    main()