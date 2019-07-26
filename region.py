import cv2
import numpy as np

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
    print("mask channel: ",regionOut.shape)
    # cv2.imshow('region',regionOut)
    # cv2.waitKey(0)
    cv2.imwrite("./Template/bin_mask/bin_mask_{}.png".format("%02d"%i),regionOut)
    return regionOut
    


