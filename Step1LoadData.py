# Dataset name: Academic Montgomery County X-ray Set
# Dataset: https://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

Height = 256
Width = 256

path = "MontgomerySet/"
imagesPath = path + "CXR_png/*.png"
leftMaskPath = path + "ManualMask/leftMask/*.png"
rightMaskPath = path + "ManualMask/rightMask/*.png"

savePath = "data/"

print("Images in folder , left mask images , right mask images :")
listOfImages = glob.glob(imagesPath)
listOfLeftMaskImages = glob.glob(leftMaskPath)
listOfRightMaskImages = glob.glob(rightMaskPath)

print(len(listOfImages))
print(len(listOfLeftMaskImages))
print(len(listOfRightMaskImages))

# load one image and display it

img = cv2.imread(listOfImages[0], cv2.IMREAD_COLOR)
print(img.shape)

img = cv2.resize(img, (Width, Height))

left_mask = cv2.imread(listOfLeftMaskImages[0], cv2.IMREAD_GRAYSCALE)
right_mask = cv2.imread(listOfRightMaskImages[0], cv2.IMREAD_GRAYSCALE)

left_mask = cv2.resize(left_mask, (Width, Height))
right_mask = cv2.resize(right_mask, (Width, Height))

finalMask = left_mask + right_mask

cv2.imshow("img", img)
cv2.imshow("left_mask", left_mask)
cv2.imshow("right_mask", right_mask)
cv2.imshow("finalMask", finalMask)

#look at one mask
# reduce the mask size to see the values :

mask16 = cv2.resize(left_mask, (16, 16))
print(mask16)

# 0 is the background (black) and 255 is the object (white)
# lets change the values to 0 and 1

mask16[mask16 > 0] = 1
print(mask16)

# lets build the full data

allImages = []
maskImages = []

# load the train images and masks

print("start loading the train images and masks + images augmentation X3")

for imgFile , leftMask, rightMask in tqdm(zip(listOfImages, listOfLeftMaskImages, listOfRightMaskImages), total = len(listOfImages)) :

    # create the Numpy data for the images
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (Width, Height))
    img = img / 255.0
    img = img.astype(np.float32)
    allImages.append(img)

    leftMask = cv2.imread(leftMask, cv2.IMREAD_GRAYSCALE)
    rightMask = cv2.imread(rightMask, cv2.IMREAD_GRAYSCALE)

    # merge left and right masks
    mask = leftMask + rightMask
    mask = cv2.resize(mask, (Width, Height))

    mask[mask>0] = 1
    maskImages.append(mask)

allImagesNP = np.array(allImages)
maskImagesNP = np.array(maskImages)
maskImagesNP = maskImagesNP.astype(int)

print("Shapes of train images and masks :")
print(allImagesNP.shape)
print(maskImagesNP.shape)

# split the data to train and validate

from sklearn.model_selection import train_test_split
split = 0.1

train_imgs, valid_imgs = train_test_split(allImagesNP, test_size=split , random_state=42)
train_masks, valid_masks = train_test_split(maskImagesNP, test_size=split, random_state=42)

print("Shapes of train images and masks : ")
print(train_imgs.shape)
print(train_masks.shape)

print("Shapes of Validat images and masks : ")
print(valid_imgs.shape)
print(valid_masks.shape)

# save the numpy arrays :

print("Save the data :")
os.makedirs(savePath, exist_ok=True)

np.save(savePath + "Unet-Train-Lung-Images.npy", train_imgs)
np.save(savePath + "Unet-Train-Lung-Masks.npy", train_masks)

np.save(savePath + "Unet-Validate-Lung-Images.npy", valid_imgs)
np.save(savePath + "Unet-Validate-Lung-Masks.npy", valid_masks)

print("Finish save the data")

 