import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageFilter
import numpy as np
from keras.datasets import cifar100
import keras
import math
from scipy.misc import imresize, imsave
from scipy import signal
import cv2
import timeit
import csv
im = None
tk_img = None
tk_img1 = None
tk_img2 = None


def edge_detect(InputData, mask1, mask2):
    sobel1 = signal.convolve2d(InputData, mask1)
    sobel1 = sobel1[1:np.size(sobel1, 0) - 1, 1:np.size(sobel1, 1) - 1]
    sobel2 = signal.convolve2d(InputData, mask2)
    sobel2 = sobel2[1:np.size(sobel2, 0) - 1, 1:np.size(sobel2, 1) - 1]
    s = np.sqrt(sobel1 ** 2 + sobel2 ** 2)
    s = s * 255 / np.max(s)
    return s


def normal(InputData):
    m = np.zeros([InputData.shape[0], InputData.shape[1]])
    min_1 = np.min(InputData, axis=None)
    max_1 = np.max(InputData, axis=None)
    for x in range(0, InputData.shape[0], 1):
        for y in range(0, InputData.shape[1], 1):
            m[x, y] = (InputData[x, y] - min_1) * 255 / (max_1 - min_1)

    return m
def normal_SM(InputData):
    n = np.zeros([InputData.shape[0], InputData.shape[1]])
    min_1 = np.min(InputData, axis=None)
    max_1 = np.max(InputData, axis=None)
    for x in range(0, InputData.shape[0], 1):
        for y in range(0, InputData.shape[1], 1):
            n[x, y] = (InputData[x, y] - min_1) * 1 / (max_1 - min_1)

    return n

def activate_SM(inputdata):
    #inputdata=np.int8(inputdata)
    n = np.zeros([inputdata.shape[0], inputdata.shape[1]])
    n = -1*(inputdata-1) /(np.log(inputdata+np.exp(1)))
    return n

def pyramid(InputImage):
    Gaussian = np.array([[2, 13, 2], [13, 40, 13], [2, 13, 2]])
    totalbii = np.zeros([8, 2])
    bi = np.uint8(InputImage)
    width = bi.shape[0]
    height = bi.shape[1]
    sub3 = np.zeros([128, 128 * 8])
    sub3[:, 0:128] = InputImage
    for x in range(0, 7, 1):
        width = width // 2
        height = height // 2
        sub1 = imresize(bi, (width, height))
        sub1 = signal.convolve2d(sub1, Gaussian)
        sub1 = sub1[1:np.size(sub1, 0) - 1, 1:np.size(sub1, 1) - 1]
        sub1 = normal(sub1)
        bi = sub1
        bi1 = sub1
        totalbi = np.array([[bi.shape[0], bi.shape[1]]])
        totalbii[0:1, :] = np.array([[InputImage.shape[0], InputImage.shape[1]]])
        totalbii[x + 1:x + 2, :] = totalbi
        width1 = bi1.shape[0]
        height1 = bi1.shape[1]
        for y in range(0, x + 1, 1):
            width1 = width1 * 2
            height1 = height1 * 2
            sub2 = imresize(bi1, (width1, height1), interp='bilinear')
            Q = (np.size(sub2, axis=0) != totalbii[x - y][0] or np.size(sub2, axis=1) != 1)
            W = (np.mod(np.size(sub2, axis=0) - totalbii[x - y][0], 2) == 0)
            E = (np.mod(np.size(sub2, axis=1) - totalbii[x - y][0], 2) == 0)
            if Q and W and E:
                # if ((sub2.shape[0] != totalbi[x+1-y,0] or sub2.shape[1] != 1) and (np.mod(sub2.shape[0]-totalbi[x+1-y,0],1)==0) and (np.mod(sub2.shape[1]-totalbi[x+1-y,0],1)==0):
                dif1 = math.floor((np.size(sub2, 0) - totalbii[x - y, 0]) / 2)
                dif2 = math.floor((np.size(sub2, 1) - totalbii[x - y, 1]) / 2)
                sub2 = sub2[dif1:np.size(sub2, 0) - dif1, dif2:np.size(sub2, 1) - dif2]
            continue
            sub2 = (signal.convolve2d(sub2, Gaussian))
            sub2 = sub2[1:np.size(sub2, 0) - 1, 1:np.size(sub2, 1) - 1]
            sub2 = normal(sub2)
            bi1 = sub2
            # sub2=np.float64(sub2)
        sub3[:, (x + 1) * 128:(x + 2) * 128] = sub2

    return sub3


def preprocessing(Inputimage):
    Inputimage1 = imresize(Inputimage, (128, 128), interp='lanczos')
    #imsave('1.jpg', Inputimage1)
    kernel1 = np.ones((5, 5),np.float32)/25
    img1 = np.double(Inputimage1)
    img3 = cv2.filter2D(Inputimage1, -1, kernel1)
    #imsave('12311.jpg', img3)
    # img2 = img1.filter(ImageFilter.BLUR)
    # img3 = np.uint8(img3)
    # img2 = np.uint8(img2)
    r = img1[:, :, 0]
    g = img1[:, :, 1]
    b = img1[:, :, 2]

    realR = r - (b + g) / 2
    realR[realR < 0] = 0
    rr = normal(realR)

    realG = g - (b + r) / 2
    realG[realG < 0] = 0
    gg = normal(realG)

    realB = b - (r + g) / 2
    realB[realB < 0] = 0
    bb = normal(realB)

    Y = ((r + g) / 2) - b
    Y[Y < 0] = 0
    yy = normal(Y)

    totalcolor = r + g + b
    # R = r/totalcolor
    # G = g/totalcolor
    # B = b/totalcolor
    i = (totalcolor) / 3

    mask1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Sobel1 = edge_detect(r, mask1, mask2)
    Sobel2 = edge_detect(g, mask1, mask2)
    Sobel3 = edge_detect(b, mask1, mask2)
    Sobel2[Sobel1 > Sobel2] = Sobel1[Sobel1 > Sobel2]
    Sobel3[Sobel2 > Sobel3] = Sobel2[Sobel2 > Sobel3]

    
    itotal = pyramid(i)
    etotal = pyramid(Sobel3)
    rtotal = pyramid(rr)
    gtotal = pyramid(gg)
    btotal = pyramid(bb)
    ytotal = pyramid(yy)
    #imsave('itotal.jpg', itotal)
    #imsave('etotal.jpg', etotal)
    #imsave('rtotal.jpg', rtotal)
    #imsave('gtotal.jpg', gtotal)
    #imsave('btotal.jpg', btotal)
    #imsave('ytotal.jpg', ytotal)

    I1 = np.zeros([128, 128])
    I2 = np.zeros([128, 128])
    I3 = np.zeros([128, 128])
    I1 = (abs(itotal[:, 128:128 * 2] - itotal[:, 128 * 3:128 * 4]) + abs(
        itotal[:, 128:128 * 2] - itotal[:, 128 * 4:128 * 5]))
    I1 = normal(I1)

    I2 = (abs(itotal[:, 128 * 2 + 1:128 * 3 + 1] - itotal[:, 128 * 4 + 1:128 * 5 + 1]) + abs(
        itotal[:, 128 * 2 + 1:128 * 3 + 1] - itotal[:, 128 * 5 + 1:128 * 6 + 1]))
    I2 = normal(I2)

    I3 = (abs(itotal[:, 128 * 3 + 1:128 * 4 + 1] - itotal[:, 128 * 5 + 1:128 * 6 + 1]) + abs(
        itotal[:, 128 * 3 + 1:128 * 4 + 1] - itotal[:, 128 * 6 + 1:128 * 7 + 1]))
    I3 = normal(I3)

    I = (I1 + I2 + I3)
    I = normal(I)

    # CSD&N
    E1 = (abs(etotal[:, 128:128 * 2] - etotal[:, 128 * 3:128 * 4]) + abs(
        etotal[:, 128:128 * 2] - etotal[:, 128 * 4:128 * 5]))
    E1 = normal(E1)

    E2 = (abs(etotal[:, 128 * 2 + 1:128 * 3 + 1] - etotal[:, 128 * 4 + 1:128 * 5 + 1]) + abs(
        etotal[:, 128 * 2 + 1:128 * 3 + 1] - etotal[:, 128 * 5 + 1:128 * 6 + 1]))
    E2 = normal(E2)

    E3 = (abs(etotal[:, 128 * 3 + 1:128 * 4 + 1] - etotal[:, 128 * 5 + 1:128 * 6 + 1]) + abs(
        etotal[:, 128 * 3 + 1:128 * 4 + 1] - etotal[:, 128 * 6 + 1:128 * 7 + 1]))
    E3 = normal(E3)

    E = (E1 + E2 + E3)
    E = normal(E)

    RG11 = abs((rtotal[:, 128:128 * 2] - gtotal[:, 128:128 * 2]) - (
            gtotal[:, 128 * 2 + 1:128 * 3 + 1] - rtotal[:, 128 * 2 + 1:128 * 3 + 1]))
    RG11 = normal(RG11)
    RG12 = abs((rtotal[:, 128:128 * 2] - gtotal[:, 128:128 * 2]) - (
            gtotal[:, 128 * 3 + 1:128 * 4 + 1] - rtotal[:, 128 * 3 + 1:128 * 4 + 1]))
    RG12 = normal(RG12)
    RG1 = RG11 + RG12
    RG1 = normal(RG1)

    BY11 = abs((btotal[:, 128:128 * 2] - ytotal[:, 128:128 * 2]) - (
            ytotal[:, 128 * 2 + 1:128 * 3 + 1] - btotal[:, 128 * 2 + 1:128 * 3 + 1]))
    BY11 = normal(BY11)
    BY12 = abs((btotal[:, 128:128 * 2] - ytotal[:, 128:128 * 2]) - (
            ytotal[:, 128 * 3 + 1:128 * 4 + 1] - btotal[:, 128 * 3 + 1:128 * 4 + 1]))
    BY12 = normal(BY12)
    BY1 = BY11 + BY12
    BY1 = normal(BY1)

    C1 = RG1 + BY1
    C1 = normal(C1)

    RG21 = abs((rtotal[:, 128 * 2 + 1:128 * 3 + 1] - gtotal[:, 128 * 2 + 1:128 * 3 + 1]) - (
            gtotal[:, 128 * 4 + 1:128 * 5 + 1] - rtotal[:, 128 * 4 + 1:128 * 5 + 1]))
    RG21 = normal(RG21)
    RG22 = abs((rtotal[:, 128 * 2 + 1:128 * 3 + 1] - gtotal[:, 128 * 2 + 1:128 * 3 + 1]) - (
            gtotal[:, 128 * 5 + 1:128 * 6 + 1] - rtotal[:, 128 * 5 + 1:128 * 6 + 1]))
    RG22 = normal(RG22)
    RG2 = RG21 + RG22
    RG2 = normal(RG2)

    BY21 = abs((btotal[:, 128 * 2 + 1:128 * 3 + 1] - ytotal[:, 128 * 2 + 1:128 * 3 + 1]) - (
            ytotal[:, 128 * 4 + 1:128 * 5 + 1] - btotal[:, 128 * 4 + 1:128 * 5 + 1]))
    BY21 = normal(BY21)
    BY22 = abs((btotal[:, 128 * 2 + 1:128 * 3 + 1] - ytotal[:, 128 * 2 + 1:128 * 3 + 1]) - (
            ytotal[:, 128 * 5 + 1:128 * 6 + 1] - btotal[:, 128 * 5 + 1:128 * 6 + 1]))
    BY22 = normal(BY22)
    BY2 = BY21 + BY22
    BY2 = normal(BY2)

    C2 = RG2 + BY2
    C2 = normal(C2)

    RG31 = abs((rtotal[:, 128 * 3 + 1:128 * 4 + 1] - gtotal[:, 128 * 3 + 1:128 * 4 + 1]) - (
            gtotal[:, 128 * 5 + 1:128 * 6 + 1] - rtotal[:, 128 * 5 + 1:128 * 6 + 1]))
    RG31 = normal(RG31)
    RG32 = abs((rtotal[:, 128 * 3 + 1:128 * 4 + 1] - gtotal[:, 128 * 3 + 1:128 * 4 + 1]) - (
            gtotal[:, 128 * 6 + 1:128 * 7 + 1] - rtotal[:, 128 * 6 + 1:128 * 7 + 1]))
    RG32 = normal(RG32)
    RG3 = RG31 + RG32
    RG3 = normal(RG3)

    BY31 = abs((btotal[:, 128 * 3 + 1:128 * 4 + 1] - ytotal[:, 128 * 3 + 1:128 * 4 + 1]) - (
            ytotal[:, 128 * 5 + 1:128 * 6 + 1] - btotal[:, 128 * 5 + 1:128 * 6 + 1]))
    BY31 = normal(BY31)
    BY32 = abs((btotal[:, 128 * 3 + 1:128 * 4 + 1] - ytotal[:, 128 * 3 + 1:128 * 4 + 1]) - (
            ytotal[:, 128 * 6 + 1:128 * 7 + 1] - btotal[:, 128 * 6 + 1:128 * 7 + 1]))
    BY32 = normal(BY32)
    BY3 = BY31 + BY32
    BY3 = normal(BY3)

    C3 = RG3 + BY3
    C3 = normal(C3)

    C = (C1 + C2 + C3)
    C = normal(C)

    SM = 0.3 * I + 0.4 * E + 0.3 * C
    SM = normal_SM(SM)
  #  uy=np.mean(SM)
  #  uy1 = np.var(SM)
  #  uy2= np.std(SM)
  #  print(uy)
  #  print(uy1)
  #  print(uy2)
    SM_A=activate_SM(SM)

    #imsave('Itensity_feature_map.jpg', I)
    #imsave('Edge_feature_map.jpg', E)
   # imsave('Color_feature_map.jpg', C)
  #  imsave('2.jpg', SM)
 #   imsave('3.jpg', SM_A)

    # img3 = Inputimage1.filter(ImageFilter.BLUR)
    # img3 = np.uint8(img33)
    # ha = np.ones((10,10), dtype = float) / 100
    # img3 = scipy.ndimage.convolve(img1,ha, mode ='symmetric')

    # ha1 = np.ones((5,5), dtype = float) / 25
    # img2 = scipy.ndimage.convolve(input0,ha1, mode ='symmetric')
    # img2 = Inputimage1.filter(ImageFilter.BLUR)
    # img2 = np.uint8(img22)

    tmp_SM_A = np.zeros((128, 128))
 #   tmp_SM2 = np.zeros((128, 128))

    SIZE1 = np.array([128, 128])
    out_img = np.ones((128, 128, 3))
    for m in range(0, SIZE1[0]):
        for n in range(0, SIZE1[1]):
            if SM_A[m, n] <=0.5:
                tmp_SM_A[m, n] = 1




 #   tmp_SM3 = tmp_SM2 - tmp_SM1
    #tmp_SM4 = tmp_SM2 + tmp_SM3


    for o in range(0, 3):
        for m in range(0, SIZE1[0]):
         for n in range(0, SIZE1[1]):
            if tmp_SM_A[m, n] == 1:
                out_img[m, n, o] = Inputimage1[m, n, o]
            else:
                out_img[m, n, o] = img3[m, n, o]

    out_images = np.uint8(out_img)
    #imsave('4.jpg', tmp_SM_A)
   # imsave('tmp_SM4.jpg', tmp_SM4)
    #    imsave('img2.jpg', img2)
    # imsave('img3.jpg', img3)
    #imsave('5.jpg', out_images)
    return out_images


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(np.shape(y_test))
#post_data = preprocessing(x_train[1, :, :, :])
#plt.imshow(post_data)
#plt.show()
start = timeit.default_timer()
#for h in range(0,10000):
#ak1 = np.zeros((33))
f=open('cif100_test.csv',newline='')
wr=csv.writer(f)
for l in range(0,100):
 post_data = preprocessing(x_train[l, :, :, :])
 imsave('out_'+str(l)+'.jpg', post_data)
 #wr.writerow(y_test[l,1])
# ak1[l]=uy
f.close()
stop = timeit.default_timer()
#plt.imshow(ak1)
plt.show()

print(stop - start)
# plt.imshow(x_test[1,:,:])
# plt.imshow(y_test[:,:])
# plt.show()
