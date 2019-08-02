from skimage import morphology,draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# #创建一个二值图像用于测试
# image = np.zeros((400, 400))

# #生成目标对象1(白色U型)
# image[10:-10, 10:100] = 1
# image[-100:-10, 10:-10] = 1
# image[10:-10, -100:-10] = 1

# #生成目标对象2（X型）
# rs, cs = draw.line(250, 150, 10, 280)
# for i in range(10):
#     image[rs + i, cs] = 1
# rs, cs = draw.line(10, 150, 250, 280)
# for i in range(20):
#     image[rs + i, cs] = 1

# #生成目标对象3（O型）
# ir, ic = np.indices(image.shape)
# circle1 = (ic - 135)**2 + (ir - 150)**2 < 30**2
# circle2 = (ic - 135)**2 + (ir - 150)**2 < 20**2
# image[circle1] = 1
# image[circle2] = 0

# #实施骨架算法
# skeleton =morphology.skeletonize(image)

# #显示结果
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('original', fontsize=20)

# ax2.imshow(skeleton, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('skeleton', fontsize=20)

# fig.tight_layout()
# plt.show()

image = Image.open("./0_denoise.png")
image = image.convert("L")
image_np = np.asarray(image)
image_np = np.clip(image_np,0,1)


#实施骨架算法
skeleton =morphology.skeletonize(image_np)
skel, distance =morphology.medial_axis(image_np, return_distance=True)

#显示结果
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=20)

ax2.imshow(skel, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()