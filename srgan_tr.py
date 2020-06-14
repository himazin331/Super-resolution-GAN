# -*- coding: utf-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import tensorflow.keras.layers as kl
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import backend as K
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
                Pixel_shuffer(128) for _ in range(2)
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
    def __init__(self, out_ch):
        super().__init__()

        input_shape = (128, 128, 64)

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
        self.vgg = VGG16(include_top=False, input_tensor=input_tensor)
        self.vgg.trainable = False
        self.vgg.outputs = [self.vgg.layers[9].output]  # VGG16 block3_conv3 output  

        # Content Loss Model
        self.cl_model = tf.keras.Model(input_tensor, self.vgg.outputs)

        # Discriminator
        discriminator_ = Discriminator()
        inputs = tf.keras.Input(shape=hr_shape)
        outputs = discriminator_(inputs)
        self.discriminator = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'])

        # Generator
        self.generator = Generator()
                   
        # Combined Model setup
        hr_input = tf.keras.Input(shape=hr_shape)
        lr_input = tf.keras.Input(shape=lr_shape)
        sr_output = self.generator(lr_input)

        self.discriminator.trainable = False
        d_fake = self.discriminator(sr_output)

        self.gan = tf.keras.Model(inputs=lr_input, outputs=[sr_output, d_fake])
        self.gan.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=[self.Content_loss, tf.keras.losses.BinaryCrossentropy()],
                        loss_weights=[1., 1e-3])
        
  
    def Content_loss(self, hr_img, sr_img):
        return K.mean(K.abs(K.square(self.cl_model(hr_img) - self.cl_model(sr_img))))

    def psnr(self, sr_img, hr_img):
        
        return cv2.PSNR(sr_img, hr_img)
        
    def train(self, lr_imgs, hr_imgs, out_path, batch_size, epoch):

        h_batch = int(batch_size / 2)

        real_lab = np.ones((h_batch, 1))  # High-resolution image label
        fake_lab = np.zeros((h_batch, 1)) # Super-resolution image label(Discriminator side)
        gan_lab = np.ones((h_batch, 1))
        
        
        # train run
        for epoch in range(epoch):

            # - Train Discriminator -

            # High-resolution image random pickups
            idx = np.random.randint(0, hr_imgs.shape[0], h_batch)
            hr_img = hr_imgs[idx]

            # Low-resolution image random pickups
            lr_img = lr_imgs[idx]

            # Discriminator enabled train
            self.discriminator.trainable = True

            # train by High-resolution image
            d_real_loss = self.discriminator.train_on_batch(hr_img, real_lab)

            # train by Super-resolution image
            sr_img = self.generator.predict(lr_img) 
            d_fake_loss = self.discriminator.train_on_batch(sr_img, fake_lab)
            d_fake = self.discriminator.predict(sr_img)
            
            # Discriminator average loss 
            d_loss = 0.5 * np.add(d_real_loss, d_fake_loss)
            

            # - Train Generator -

            # High-resolution image random pickups
            idx = np.random.randint(0, hr_imgs.shape[0], h_batch)
            hr_img = hr_imgs[idx]

            # Low-resolution image random pickups
            lr_img = lr_imgs[idx]


            self.discriminator.trainable = False

            g_loss = self.gan.train_on_batch(lr_img, [hr_img, gan_lab])

            # Epoch num
            print("Epoch: {} D_loss: {} G_loss: {} PSNR: {}".format(epoch+1, d_loss, g_loss, self.psnr(sr_img, hr_img)))

        # Parameter-File Saving
        self.generator.save_weights(out_path)

def create_dataset(data_dir, h, w, mag):

    lr_imgs = []
    hr_imgs = []

    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)

        _, ext = os.path.splitext(c)
        if ext.lower() != '.bmp':
            continue

        img = cv2.imread(d)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (h, w))

        img_low = cv2.resize(img, (int(h/mag), int(w/mag)))
        img_low = cv2.resize(img_low, (h, w))

        lr_imgs.append(img_low)
        hr_imgs.append(img)

    lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32) 
    lr_imgs = (lr_imgs.numpy() - 127.5) / 127.5
 

    hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
    hr_imgs = (hr_imgs.numpy() - 127.5) / 127.55
    
    return lr_imgs, hr_imgs

def main():

    # プログラム情報
    print("Super-resolution GAN training ver.3")
    print("Last update date:    2020/05/11\n")
    
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='Super-resolution GAN training')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='画像フォルダパスの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='パラメータの保存先指定(デフォルト値=./srcnn.h5')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='ミニバッチサイズの指定(デフォルト値=32)')
    parser.add_argument('--epoch', '-e', type=int, default=3000,
                        help='学習回数の指定(デフォルト値=3000)')
    parser.add_argument('--he', '-he', type=int, default=128,
                        help='リサイズの高さ指定(デフォルト値=128)')      
    parser.add_argument('--wi', '-wi', type=int, default=128,
                        help='リサイズの指定(デフォルト値=128)')
    parser.add_argument('--mag', '-m', type=int, default=2,
                        help='縮小倍率の指定(デフォルト値=2)')                           
    args = parser.parse_args()

    # 画像フォルダパス未指定->例外
    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    # 存在しない画像フォルダ指定時->例外
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder \"{}\" is not found.\n".format(args.data_dir))
        sys.exit()

    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "srgan.h5")

    # 設定情報出力
    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(out_path))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("")
    print("# Height: {}".format(args.he))
    print("# Width: {}".format(args.wi))
    print("# Magnification: {}".format(args.mag))
    print("===========================\n")

    lr_imgs, hr_imgs = create_dataset(data_dir, args.he. args.wi, args.mag)

    Trainer = trainer()
    Trainer.train(lr_imgs, hr_imgs, out_path=out_path, batch_size=batch_size, epoch=epoch)

if __name__ == '__main__':
    main()