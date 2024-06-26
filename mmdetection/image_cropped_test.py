#coding:utf-8
import cv2
import os
import codecs

#overlap_half=True滑动窗口切图，每次有一半区域重叠，这时候x方向的步长就是窗口宽度的一半，y方向的步长是窗口高度的一半，stridex和stridey参数将不再起作用
def slide_crop(img, kernelw, kernelh, overlap_half=True, stridex=0, stridey=0):
    height, width, _ = img.shape
    if overlap_half:
        stridex = int(kernelw / 2)
        stridey = int(kernelh / 2)
    img_list = []
    corner_list = []
    stepx = int(width / stridex)
    stepy = int(height / stridey)
    
    for r in range(stepy-1): #stepy-1
        starty = r * stridey
        for c in range(stepx-1):
            startx = c * stridex
            corner_list.append((startx,starty))
            img_list.append(img[starty:starty+kernelh, startx:startx+kernelw,:])
            
    if (width / stridex) % 1 > 0.01:
        print('x',(width / stridex))
        for r in range(stepy-1):
            starty = r * stridey
            startx = width - kernelw
            corner_list.append((startx,starty))
            img_list.append(img[starty:starty+kernelh, startx:startx+kernelw,:])
    if (height / stridey) % 1 > 0.01:
        print(height,stridey,(height / stridey))
        for c in range(stepx-1):
            startx = c * stridex
            starty = height - kernelh
            corner_list.append((startx,starty))
            img_list.append(img[starty:starty+kernelh, startx:startx+kernelw,:])
    if (width / stridex) % 1 > 0.01 and (height / stridey) % 1 > 0.01:
        starty = height - kernelh
        startx = width - kernelw
        corner_list.append((startx,starty))
        img_list.append(img[starty:starty+kernelh, startx:startx+kernelw,:])
    return img_list,corner_list

def crop_dataset(imgpath, windows_width, windows_height, annotation, savePath):

    if os.path.exists(savePath) is False:
        os.mkdir(savePath)
    
    count = 0
    name = os.path.basename(imgpath)[:-4]#'20240420_bg'
    
    img = cv2.imread(os.path.join(imgpath))
    height = img.shape[0]
    width = img.shape[1]
    croped_width = windows_width
    croped_height = windows_height
    count += 1

    img_list, corner_list = slide_crop(img, croped_width, croped_height, overlap_half=True)

    for num,img in enumerate(img_list):
        saveImgPath = savePath + '/' + '{}_{}.jpg'.format(name, num)
        print(saveImgPath)
        cv2.imwrite(saveImgPath, img)

if __name__ == '__main__':
    windows_width = 3000 
    windows_height = 1220 
    imgpath = './fusion_image/val'
    annotation = './fusion_image/val/val.txt'
    savePath = './fusion_image/val/images'
    
    crop_dataset(imgpath, windows_width, windows_height, annotation, savePath)
