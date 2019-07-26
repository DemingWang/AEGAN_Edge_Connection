import cv2
import numpy as np
import glob
from PIL import Image

def regionGenerate(img):
    findcontourimg = img.copy()
    # if img.ndim < 3:
    #     findcontourimg = np.expand_dims(findcontourimg,axis=2)
    findcontourimg = np.clip(findcontourimg, 0, 255)# 归一化也行
    findcontourimg = np.array(findcontourimg,np.uint8)

    ret, binary = cv2.threshold(findcontourimg,127,255,cv2.THRESH_BINARY)
    contours,hierarchy= cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    regionOut = np.zeros((binary.shape[0],binary.shape[1]),np.uint8)*255

    isHoleExist = True
    c_max = []
    c_min = []
    lenContours = len(contours)
    if lenContours == 2:
        isHoleExist = False
    #print(lenContours)
    hierarchy0 = hierarchy[0]
    for i in range(lenContours):
        hierarchyI = hierarchy0[i]
        if  hierarchyI[3] == -1: #hierarchyI[0] == -1 and hierarchyI[1] == -1 and
            cnt = contours[i]
            c_max.append(cnt)
        if  hierarchyI[2] == -1:#hierarchyI[0] == -1 and hierarchyI[1] == -1 and
            cnt = contours[i]
            c_min.append(cnt)
    cv2.drawContours(regionOut, c_max, -1,  (255,255,255), cv2.FILLED)
    if isHoleExist:
        cv2.drawContours(regionOut, c_min, -1,  (0,0,0), cv2.FILLED)
        cv2.drawContours(regionOut, c_min, -1,  (255,255,255), 1)
    print("mask channel: ",regionOut.shape)
    # cv2.imshow('region',regionOut)
    # cv2.waitKey(0)
    # cv2.imwrite("./Template/bin_mask/bin_mask_{}.png".format("%02d"%i),regionOut)
    return regionOut
    
if __name__ == '__main__':
    img_dir = ("./Template/bin_contour/")
    img_files = glob.glob(img_dir + "*.png")
    img_files.sort(key=lambda x:int(x[-6:-4]))
    template_imglist = []
    for tempID,img_file in enumerate(img_files):
        template_img = Image.open(img_file)
        template_img = template_img.convert("L")
        data = np.asarray(template_img)
        data = regionGenerate(data)
        sub_img = template_img - data
        cv2.imshow('sub',sub_img)
        cv2.waitKey(0)

