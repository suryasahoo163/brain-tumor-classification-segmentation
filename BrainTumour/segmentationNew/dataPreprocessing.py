import numpy as np
import os,sys, cv2
import nibabel as nb
import matplotlib.pyplot as plt

cwd = os.getcwd()
os.chdir(cwd)

newDataDir = os.path.join(cwd,'NewData')
# os.mkdir(newDataDir)

dataDir = os.path.join(cwd,r'Data\BraTS2024_small_dataset')

testDir = os.listdir(dataDir)[100]
testDir = os.path.join(dataDir,testDir)

images =[]
for i in os.listdir(testDir):
    i = os.path.join(testDir,i)
    image = nb.load(i)
    image = image.get_fdata()
    images.append(image)
images = np.array(images)
volume = images[0]
num_slices = 4
z_indices = np.linspace(0, volume.shape[2] - 1, num_slices, dtype=int)
v =(volume[:,:,z_indices[1]])

cv2.imshow('frame',v.T)
cv2.waitKey(0)


sys.exit()
cv2.imshow("frame",images)
cv2.waitKey(0)


plt.subplots(2,2,figsize = (10,10))
for i in images:
    plt.imshow(image[i])
plt.show()