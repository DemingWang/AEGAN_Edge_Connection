# 利用分类器识别类别，进而选择相应的模型
# 利用加噪声的图像进行降噪自编码器的测试
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
#from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
#from SSIM_PIL import compare_ssim as ssim
#import pytorch_ssim
import os
from imgaug import augmenters as iaa
import cv2

TestFlag = 1

if not os.path.exists('../Test_Image'):
    os.mkdir('../Test_Image')
if not os.path.exists('../Test_Image/input'):
    os.mkdir('../Test_Image/input')
if not os.path.exists('../Test_Image/output'):
    os.mkdir('../Test_Image/output')
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

x = []
Testfolder = False
#如果是读取某一个文件夹里的
if(Testfolder):
    img_dir = ("../Test_Image/input/")
    img_files = glob.glob(img_dir + "*.png")
    img_files.sort(key=lambda x:int(x[-6:-4]))
    print(img_files)

    # Load Image
    # AutoEncoder does not have to label data

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
else:
    #如果是读取DefectDataset/noise文件夹下的某些图片
    img_dir = ("../DefectDataset/Single/noise/")
    for tempID in range(0,27):
        selectNum = 0
        img_filename = img_dir+"temp_{}_{}.png".format("%02d"%tempID,"%04d"%selectNum)
        print("Image Filename: ",img_filename)
        img = Image.open(img_filename)
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

#AEGenerator_SK == AutoEncoderForGeneratorWithSmallKernel(3*3 Kernel),同时修改激活函数为:nn.LeakyRelu(0.2,True)
class AEGenerator_SK(nn.Module):
    def __init__(self):
        super(AEGenerator_SK, self).__init__()
        self.encoder = nn.Sequential( #input 1*256*256
            nn.Conv2d(1,32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 32*128*128
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32,32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 32*64*64
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32,64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 64*32*32
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64,64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 64*16*16
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64,128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2,True),# 128*8*8
            nn.MaxPool2d((2,2))
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
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 16 * 16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 256 * 32 * 32

            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 128 * 64 * 64
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True), # 64 * 128 * 128

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),  # b, 16, 5, 5
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

class EncoderWithClassifier(nn.Module):
    def __init__(self):
        super(EncoderWithClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, 5, stride=2, padding=2),
            nn.ReLU(True),# 64*128*128

            nn.Conv2d(32,32, 5, stride=2, padding=2),
            nn.ReLU(True),# 32*64*64

            nn.Conv2d(32,64, 5, stride=2, padding=2),
            nn.ReLU(True),# 64*32*32

            nn.Conv2d(64,64, 5, stride=2, padding=2),
            nn.ReLU(True),# 64*16*16

            nn.Conv2d(64,128, 5, stride=2, padding=2),
            nn.ReLU(True)# 128*8*8
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*8*8, 128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,27),
            nn.Softmax()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128,101),
            nn.ReLU(True)
        )
        self.relu = nn.ReLU(False)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        label = self.fc2(x)
        code = self.fc3(x)
        x = torch.cat((label,code),1)
        return x,label


class Decoder(nn.Module): ##这里的网络结构实际上是参考了Implicit3D的结构
    def __init__(self):
        super(Decoder, self).__init__()
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
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)
        return x
 
model_e = EncoderWithClassifier().cuda()
model_g = Decoder().cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Model()
#if torch.cuda.device_count() > 1:
model_e = nn.DataParallel(model_e,device_ids=[0])
model_g = nn.DataParallel(model_g,device_ids=[0])

model_e.to(device)
model_g.to(device)

# model.load_state_dict(torch.load('./model/aug/conv_aae_epoch_2990.pth'))
 
checkpoint_e = torch.load('../log/model/20190801/encoder/aegan_epoch_285.pth')
# here, checkpoint is a dict with the keys you defined before
model_e.load_state_dict(checkpoint_e['model'])


checkpoint_g = torch.load('../log/model/20190801/decoder/aegan_epoch_285.pth')
# here, checkpoint is a dict with the keys you defined before
model_g.load_state_dict(checkpoint_g['model'])


batch_test=torch.Tensor(x_test)
img = Variable(batch_test).cuda()
# ===================forward=====================

z_code,label_pre = model_e(img)
print(type(label_pre))
np_label_pres = label_pre.cpu().data.numpy()
labeloutput = [np.argmax(np_label_pre)for np_label_pre in np_label_pres]
print(labeloutput)


model_ae = AEGenerator().cuda()
model_ae = nn.DataParallel(model_ae,device_ids=[0])
model_ae.to(device)

for idx, label_pre in enumerate(labeloutput):
    checkpoint_ae = torch.load('../Model_Multi/model/{}/{}_aegan_epoch_199.pth'.format("%02d"%idx,"%02d"%idx))
    # here, checkpoint is a dict with the keys you defined before
    model_ae.load_state_dict(checkpoint_ae['model'])
    singleimg = img[idx].unsqueeze(0)
    outputsingleimg = model_ae(singleimg)

    outputsingleimgnp = outputsingleimg.cpu().data.numpy()
    singleimgnp = singleimg.cpu().data.numpy()

    outputsingleimgnp = outputsingleimgnp * 255
    outputsingleimgnp = outputsingleimgnp.transpose(0,2,3,1)
    outputsingleimgnp = outputsingleimgnp.squeeze(0)

    singleimgnp = singleimgnp * 255
    singleimgnp = singleimgnp.transpose(0,2,3,1)
    singleimgnp = singleimgnp.squeeze(0)
    print("singleimgnp.shape: ",singleimgnp.shape)
    cv2.imshow("test!!!!!",singleimgnp)
    cv2.waitKey(0)
    cv2.imwrite("../Test_Image/output/{}_noise_de.png".format(idx),outputsingleimgnp)
    cv2.imwrite("../Test_Image/output/{}_noise.png".format(idx),singleimgnp)


# output = model_g(z_code)
# output_imgs = output.cpu().data.numpy()
# noise_imgs = img.cpu().data.numpy()
# # print(type(output_img))
# # print(output_img.shape)

# output_imgs = output_imgs * 255
# output_imgs = output_imgs.transpose(0,2,3,1)

# noise_imgs = noise_imgs * 255
# noise_imgs = noise_imgs.transpose(0,2,3,1)

# for i,singleimg in enumerate(output_imgs):

#     cv2.imwrite("../Test_Image/output/{}_noise_de.png".format(i),singleimg)
#     cv2.imwrite("../Test_Image/output/{}_noise.png".format(i),noise_imgs[i])
    

 

