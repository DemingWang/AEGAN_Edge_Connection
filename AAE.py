
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
 
import matplotlib.gridspec as gridspec
import os
from PIL import Image

if not os.path.exists('./GAN_Image'):
    os.mkdir('./GAN_Image')

if not os.path.exists('./Model'):
    os.mkdir('./Model')

if not os.path.exists('./Model/GAN'):
    os.mkdir('./Model/GAN')

if not os.path.exists('./Model/DIS'):
    os.mkdir('./Model/DIS')


class DefectDataset(Dataset):
  
    def __init__(self, sample_root, label_root, height, width, augment=None):
        sfiles = [x.path for x in os.scandir(sample_root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        sfiles.sort(key=lambda x:int(x[-8:-4]))
        self.sample_files = np.array(sfiles)
        lfiles = [x.path for x in os.scandir(label_root) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")]
        lfiles.sort(key=lambda x:int(x[-8:-4]))
        self.label_files = np.array(lfiles)
        self.height = height
        self.width = width
        self.augment = augment   # 是否需要图像增强

    def __len__(self):
        
        return len(self.sample_files)
 
    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        label_path = self.label_files[idx]

        img = Image.open(sample_path)
        img = img.convert("L")
        if(img.width != self.width or img.height != self.height):
            img = img.resize((self.width, self.height), 1)
        data = np.asarray(img)
        data = data.astype('float32') / 255.
        data = np.expand_dims(data,axis=0)
        sample = data.copy()
        sample = torch.from_numpy(sample)

        img = Image.open(label_path)
        img = img.convert("L")
        if(img.width != self.width or img.height != self.height):
            img = img.resize((self.width, self.height), 1)
        img = img.resize((self.width, self.height), 1)
        data = np.asarray(img)
        data = data.astype('float32') / 255.
        data = np.expand_dims(data,axis=0)
        label = data.copy()
        label = torch.from_numpy(label)

        # if self.augment:
        #     sample = self.augment(sample)
        #     label = self.augment(label)

        # print("label shape: ",label.shape)
        return sample, label     # 将读取到的图像变成tensor再传出


def showimg(images,count):
    images=images.to('cpu')
    images=images.detach().numpy()
    images=images[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]]
    images=255*(0.5*images+0.5)
    images = images.astype(np.uint8)
    grid_length=int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4,4))
    width = images.shape[2]
    gs = gridspec.GridSpec(grid_length,grid_length,wspace=0,hspace=0)
    print(images.shape)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width,width),cmap = plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
#     print('showing...')
    plt.tight_layout()
    plt.savefig('./GAN_Image/{}.png'.format(count), bbox_inches = 'tight')

def loadMNIST(batch_size):  #MNIST图片的大小是28*28
    trans_img=transforms.Compose([transforms.ToTensor()])
    trainset=MNIST('./data',train=True,transform=trans_img,download=True)
    testset=MNIST('./data',train=False,transform=trans_img,download=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset,testset,trainloader,testloader

class AEGenerator(nn.Module):
    def __init__(self):
        super(AEGenerator, self).__init__()
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


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1,16,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)), # 16 * 128 * 128

            nn.Conv2d(16,16,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)),# 16 * 64 * 64

            nn.Conv2d(16,32,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)), # 64 * 32 * 32

            nn.Conv2d(32,32,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)), # 64 * 16 * 16

            nn.Conv2d(32,64,5,stride=1,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)), # 64 * 8 * 8
        )

        self.fc = nn.Sequential(
            nn.Linear(8*8*64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class generator(nn.Module):
    def __init__(self, input_size, num_features):
        super(generator,self).__init__()
        self.fc = nn.Linear(input_size,num_features)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1,50,3,stride=1,padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),

            nn.Conv2d(50,25,3,stride=1,padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),

            nn.Conv2d(25,1,2,stride=2),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.size(0),1,56,56)
        x = self.br(x)
        x = self.gen(x)
        return x

 # Setting Image Propertie

width = 256
height = 256
pixels = width * height * 1  # gray scale

initEpoch = 260
num_epochs = 3000
num_gepochs = 5
batch_size = 100
learning_rate = 1 * 1e-4
useFineTune = True


if __name__ == "__main__":
    count = 0
    dataset = DefectDataset('./DefectDataset/noise', './DefectDataset/gt', width, height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ae_criterion = nn.BCELoss()
    d_criterion = nn.BCELoss()

    D = discriminator()  
    G = AEGenerator()

    D = D.cuda()
    G = G.cuda()


    summary(D,(1,256,256))
    summary(G,(1,256,256))

    d_optimizer = optim.Adam(D.parameters(),lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

    saved_dict_G = {
        'model': G.state_dict(),
        'opt': g_optimizer.state_dict()
    }

    saved_dict_D = {
        'model': D.state_dict(),
        'opt': d_optimizer.state_dict()
    }


    for i in range(num_epochs):
        # for (img, label) in trainloader:
        for (noise_img, gt_img) in dataloader:
            
            noise_img = Variable(noise_img).cuda()
            gt_img = Variable(gt_img).cuda()

            """ Update Discriminator """ 

            real_label = Variable(torch.ones(batch_size,1)).cuda()
            fake_label = Variable(torch.zeros(batch_size,1)).cuda()

            real_out = D(gt_img)
            d_loss_real = d_criterion(real_out,real_label) ### d_loss_real = log(D(x))
            real_scores = real_out

            fake_img = G(noise_img)
            fake_out = D(fake_img)
            d_loss_fake = d_criterion(fake_out,fake_label) ### d_loss_fake = log(1-D(G(x~)))
            fake_scores = fake_out

            d_loss = d_loss_real + d_loss_fake ### d_loss = d_loss_real + d_loss_fake = log(D(x)) + log(1-D(G(x~)))
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # noise_img, gt_img = data
            """ Update AutoEncoder """ #先进行Autoencoder的训练
            for j in range(num_gepochs):               
                # fake_label = Variable(torch.ones(batch_size)).cuda()
                # z = Variable(torch.randn(num_img,z_dimension)).cuda()
                
                fake_img = G(noise_img)
                g_loss = ae_criterion(fake_img,gt_img)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                i, num_epochs, d_loss.data, g_loss.data,
                real_scores.data.mean(), fake_scores.data.mean()))

        torch.save(saved_dict_G, './Model/GAN/aegan_epoch_{}.pth'.format(i))
        torch.save(saved_dict_D, './Model/DIS/aegan_epoch_{}.pth'.format(i))
        
        showimg(fake_img,count)
        # plt.show()
        count += 1

            
