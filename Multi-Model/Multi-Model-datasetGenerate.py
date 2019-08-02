#根据模板图片生成数据集
from imgaug import augmenters as iaa
import cv2
import glob
from PIL import Image
import numpy as np
import os
import region 
import math

transformNum = 600

if not os.path.exists('../Template'):
    os.mkdir('../Template')

if not os.path.exists('../DefectDataset'):
    os.mkdir('../DefectDataset')

if not os.path.exists('../DefectDataset/gt'):
    os.mkdir('../DefectDataset/gt')

if not os.path.exists('../DefectDataset/noise'):
    os.mkdir('../DefectDataset/noise')

if not os.path.exists('../DefectDataset/background'):
    os.mkdir('../DefectDataset/background')


def calSobel(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img = cv2.erode(img, kernel)

    sobelx = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=3)
    sobelx=cv2.convertScaleAbs(sobelx)#转回uint8
    sobely=cv2.convertScaleAbs(sobely)

    dst=cv2.addWeighted(sobelx,1,sobely,1,0)

    dst = np.clip(dst,0,255)

    # dst = cv2.erode(dst, kernel)
    if dst.ndim < 3:
        dst = np.expand_dims(dst,axis=2)
    # cv2.imshow('sobelx',sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('sobely',sobely)
    # cv2.waitKey(0)
    # cv2.imshow('dst',dst)
    # cv2.waitKey(0)
    # print('sobletype',sobelx.dtype)
    # print('sobletype',sobely.dtype)

    return dst
# 对噪声图片进行旋转、放缩、平移和dropout
seqNoise = iaa.Sequential([
    iaa.Affine(translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
        rotate=(-180, 180),
        scale=(0.8, 1.2)
    )
    # iaa.CoarseDropout((0.4, 0.8), size_percent=(0.01,0.2))
])

#平移变换 对最初的模板图片就行随机的旋转、放缩和平移
seqAffine = iaa.Sequential([
    iaa.Affine(translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)}
        # rotate=(-5, 5),
        # scale=(0.95, 1.05)
    )
])

#dropout变换 对模板图片进行黑色的dropout操作
seq_drop_gt = iaa.Sequential([
    # iaa.Fliplr(0.5), # 0.5 is the probability, horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    # iaa.Invert(0.5),
    iaa.CoarseDropout((0.1, 0.3), size_percent=(0.1,0.2))
])
#dropout变换 对模板的mask进行dropout，反色后相当于白色的噪声
seq_drop_mask = iaa.Sequential([
    # iaa.Fliplr(0.5), # 0.5 is the probability, horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    # iaa.Invert(0.5),
    iaa.CoarseDropout((0.1, 0.3), size_percent=(0.1,0.2))
])

# Own Your Image Directory
img_dir = ("../Template/bin_contour/")
img_files = glob.glob(img_dir + "*.png")
img_files.sort(key=lambda x:int(x[-6:-4]))
print("template num:", len(img_files))

# 读取背景噪声图片并进行扩充

noise_img_dir = ("../DefectDataset/background/")
noise_img_files = glob.glob(noise_img_dir + "*.png")
noise_img_files.sort(key=lambda x:int(x[-6:-4]))
#print(noise_img_files)

noise_img_list = []
for imgID, noise_img_file in enumerate(noise_img_files):
    noise_img = Image.open(noise_img_file)
    noise_img_np =cv2.cvtColor(np.array(noise_img),cv2.COLOR_RGB2GRAY) 
    # cv2.imshow("aug",noise_img_np)
    # cv2.waitKey(0)  
    noise_img_np = np.expand_dims(noise_img_np,axis=2)
    noise_img_list.append(noise_img_np)

noise_img_np = np.array(noise_img_list)
numOfRepeat = math.ceil(transformNum / len(noise_img_files))
noise_img_repeat = np.repeat(noise_img_np, numOfRepeat, axis=0)
noise_img_repeat = noise_img_repeat[:transformNum]
noise_img_repeat = seq_drop_mask.augment_images(noise_img_repeat)





#读取模板图片
template_imglist = []
for tempID,img_file in enumerate(img_files):
    template_img = Image.open(img_file)
    template_img = template_img.convert("L")
    # 图像resize和随机裁剪 
    # print(type(template_img))
    # print(template_img.size)
    data = np.asarray(template_img)
    # print(type(data))
    # print(data.shape)
    data = region.regionGenerate(data)
    # print("here")
    #print(type(data))
    #print(data.shape)
    data = np.expand_dims(data,axis=0)
    data = np.expand_dims(data,axis=0)
    #print(type(data))
    #print(data.shape)
    data = data.transpose(0,2,3,1)
    #print(type(data))
    #print(data.shape)
    ori_gt = np.repeat(data, transformNum, axis=0)
    ori_gt = ori_gt.astype('uint8')
    
    ## 生成全白色的mask图像，进行黑色的dropout，而后反色得到白色的噪声
    # mask_img = ori_gt.copy()
    # mask_img[mask_img != 255] = 255
    # mask_drop = seq_drop_mask.augment_images(mask_img)
    # whitenoise = 255 - mask_drop 
    pro_gt = seqAffine.augment_images(ori_gt)
    #print(type(pro_gt))
    #print(pro_gt.shape)
    pro_gt[:,0:2,:,:] = 0
    pro_gt[:,-2:,:,:] = 0
    pro_gt[:,:,0:2,:] = 0
    pro_gt[:,:,-2:,:] = 0

    # for i,singleimg in enumerate(pro_gt):
            # singleimg = singleimg.astype('uint8')
            # cv2.imshow("aug",singleimg)
            # cv2.waitKey(0)
            # print(singleimg.shape)
            # singleimg = np.squeeze(singleimg)
            # print(singleimg.shape)
            # sobelimg = calSobel(singleimg)
            # cv2.imwrite("./DefectDataset/whitenoise/temp_{}_{}.png".format(tempID,"%04d"%i),sobelimg)


    for i,singleimg in enumerate(pro_gt):
        # singleimg = singleimg.astype('uint8')
        # cv2.imshow("aug",singleimg)
        singleimg = np.squeeze(singleimg)
        sobelimg = calSobel(singleimg)
        pro_gt[i] = sobelimg.copy()
        cv2.imwrite("../DefectDataset/Mutil-Model/gt/{}/temp_{}_{}.png".format("%02d"%tempID,"%02d"%tempID,"%04d"%i),pro_gt[i])


    #生成平移变换后的图片并保存
    mid_pro_gt = pro_gt.copy()
    # mid_pro_gt[:,0:2,:,:] = 255
    # mid_pro_gt[:,-2:,:,:] = 255
    # mid_pro_gt[:,:,0:2,:] = 255
    # mid_pro_gt[:,:,-2:,:] = 255

    noise_img = seq_drop_gt.augment_images(mid_pro_gt)
    # print(type(noise_img))
    # print(noise_img.shape)
    aug_noise_img_repeat = seqNoise.augment_images(noise_img_repeat)
    thresh = (aug_noise_img_repeat < 255)
    aug_noise_img_repeat[thresh] = 0
    noise_img = noise_img + aug_noise_img_repeat
    noise_img = np.clip(noise_img,0,255)

    for i,singleimg in enumerate(noise_img):
        # singleimg = singleimg.astype('uint8')
        # cv2.imshow("aug",singleimg)
        cv2.imwrite("../DefectDataset/Mutil-Model/noise/{}/temp_{}_{}.png".format("%02d"%tempID,"%02d"%tempID,"%04d"%i),singleimg)

