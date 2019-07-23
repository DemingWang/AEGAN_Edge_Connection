#利用加噪声的图像进行降噪自编码器的测试
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

import glob
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from SSIM_PIL import compare_ssim as ssim
#import pytorch_ssim
import os
from imgaug import augmenters as iaa
import cv2

TestFlag = 1

if not os.path.exists('./Test_Image'):
    os.mkdir('./Test_Image')
if not os.path.exists('./Test_Image/input'):
    os.mkdir('./Test_Image/input')
if not os.path.exists('./Test_Image/output'):
    os.mkdir('./Test_Image/output')
 # Setting Image Propertie

width = 256
height = 256
pixels = width * height * 1  # gray scale

seq = iaa.Sequential([
    # iaa.Fliplr(0.5), # 0.5 is the probability, horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
    # iaa.Invert(0.5),
    iaa.CoarseDropout((0.1, 0.3), size_percent=(0.1,0.2))
])



img_dir = ("./Test_Image/input/")
img_files = glob.glob(img_dir + "*.png")
img_files.sort(key=lambda x:int(x[-6:-4]))
print(img_files)


# Load Image
# AutoEncoder does not have to label data
x = []
transformNum = 100
#读取图片
for i, f in enumerate(img_files):
    img = Image.open(f)
    img = img.convert("L")
    # 图像resize和随机裁剪 
    print(type(img))
    print(img.size)
    if(img.width != width or img.height != height):
            img = img.resize((width, height), 1)
    data = np.asarray(img)
    x.append(data)

x_truth = np.reshape(x, (len(x), width, height, 1))  # adapt this if using `channels_first` image data format



#先增加一个维度
 
# user_emb_dims = np.expand_dims(self.user_emb, axis=0)
# user_emb_dims.shape

x_test = x_truth


 

x_truth = np.array(x_truth)
x_truth = x_truth.astype('float32') / 255.
x_test = np.array(x_test)
x_test = x_test.astype('float32') / 255.



x_truth = np.reshape(x_truth, (len(x_truth),1, width, height))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test),1, width, height))  # adapt this if using `channels_first` image data format

print (x_test.shape)


 
class AEGenerator(nn.Module):
    def __init__(self):
        super(AEGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, 5, stride=2, padding=2),
            nn.ReLU(True),# 64*128*128

            nn.Conv2d(32,32, 5, stride=2, padding=2),
            nn.ReLU(True),# 128*64*64

            nn.Conv2d(32,64, 5, stride=2, padding=2),
            nn.ReLU(True),# 256*32*32

            nn.Conv2d(64,64, 5, stride=2, padding=2),
            nn.ReLU(True),# 256*16*16

            nn.Conv2d(64,128, 5, stride=2, padding=2),
            nn.ReLU(True)# 512*8*8
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*8*8, 128),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128 * 8 * 8),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 16 * 16

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 32 * 32

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 128 * 64 * 64

            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 64 * 128 * 128

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # b, 16, 5, 5
            nn.Sigmoid() # 1 * 256 * 256            
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)
        return x
 
model = AEGenerator().cuda()
# model.load_state_dict(torch.load('./model/aug/conv_aae_epoch_2990.pth'))
 
checkpoint = torch.load('./Model/GAN/aegan_epoch_679.pth')
# here, checkpoint is a dict with the keys you defined before
model.load_state_dict(checkpoint['model'])

batch_test=torch.Tensor(x_test)
img = Variable(batch_test).cuda()
# ===================forward=====================
output = model(img)
output_imgs = output.cpu().data.numpy()
noise_imgs = img.cpu().data.numpy()
# print(type(output_img))
# print(output_img.shape)

output_imgs = output_imgs * 255
output_imgs = output_imgs.transpose(0,2,3,1)

noise_imgs = noise_imgs * 255
noise_imgs = noise_imgs.transpose(0,2,3,1)

for i,singleimg in enumerate(output_imgs):

    cv2.imwrite("./Test_Image/output/{}_denoise.png".format(i),singleimg)
    cv2.imwrite("./Test_Image/output/{}_noise.png".format(i),noise_imgs[i])
    

 

