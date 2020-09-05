# Super-resolution-GAN
Super Resolution GAN with TF2.0

## srgan_tr.py

**Command**
```
python srgan_tr.py -d <DATA_DIR> -e <EPOCH_NUM> -b <BATCH_SIZE>
                                (-o <OUT_PATH> -he <HEIGHT> -wi <WIDTH> -m <MAG_SIZE>)
                           
EPOCH_NUM  : 1000 (Default)  
BATCH_SIZE : 32 (Default)
OUT_PATH   : ./srgan.h5 (Default)  
HEIGHT     : 128 (Default) *Input image size
WIDTH      : 128 (Default) *Input image size
MAG_SIZE   : 2 (Default) *See below for details.
```

**What are MAG_SIZE(-m)?**

The program creates low-resolution images from high-resolution images.  
Use `cv2.resize()` to resize the image to a smaller size and then restore it to its original size.  
By doing so, you can produce a low-resolution image.  
This is where MAG_SIZE comes into play. MAG_SIZE is the value that determines how much to shrink.

Like this!
```
        # Resize the width and height of the image divided by MAG_SIZE
        img_low = cv2.resize(img, (int(h/mag), int(w/mag)))
        
        # Resize to original size
        img_low = cv2.resize(img_low, (h, w))
```

## srgan_pre.py

**Command**
```
python srgan_pre.py -p <PARAM_NAME> -d <IMG_DIR>
                                (-o <OUT_PATH> -he <HEIGHT> -wi <WIDTH> -m <MAG_SIZE>)
                                
OUT_PATH   : ./result (Default)  
HEIGHT     : 128 (Default) *Input image size
WIDTH      : 128 (Default) *Input image size
MAG_SIZE   : 2 (Default) *See below for details.
```
