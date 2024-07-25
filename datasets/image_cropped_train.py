#coding:utf-8
import cv2
import os
import codecs
import tqdm

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

    for r in range(stepy): # 0~stepy-1-1
        starty = min(r * stridey, height - kernelh)
        for c in range(stepx): # 0~stepx-1-1
            startx = min(c * stridex, width - kernelw)
            corner_list.append((startx,starty))
            img_list.append(img[starty:starty+kernelh, startx:startx+kernelw,:])
    
    return img_list,corner_list

def crop_dataset(imgpath, windows_width, windows_height, srcAnn, annotation, cropAnno, savePath):

    if os.path.exists(savePath) is False:
        os.mkdir(savePath)
    
    with codecs.open(annotation,'r',encoding='utf-8') as f:
        annotationList = f.readlines()

    count = 0

    while count < len(annotationList):
        name = annotationList[count].strip('\n')
        print(name)
        img = cv2.imread(os.path.join(imgpath, name + '.jpg'))
        height = img.shape[0]
        width = img.shape[1]
        croped_width = windows_width
        croped_height = windows_height
        count += 1
        
        boxes = []
        with codecs.open(os.path.join(srcAnn, name + '.txt'), 'r', encoding='utf-8') as f:
            box_list = f.readlines()
            for line in box_list:
                point = line.split() 
                class_idx = int(point[0])
                x_center, y_center, w, h = float(point[1])*width, float(point[2])*height, float(point[3])*width, float(point[4])*height
                boxes.append([x_center, y_center, w, h])
        
        img_list, corner_list = slide_crop(img, croped_width, croped_height, overlap_half=True)
        
        boxes_list = [[] for i in range(len(corner_list))]
        
        for i, (x, y) in enumerate(corner_list):
            for box in boxes:
                x1 = round(box[0]-box[2]/2)
                y1 = round(box[1]-box[3]/2)
                x2 = round(box[0]+box[2]/2)
                y2 = round(box[1]+box[3]/2)
            
                if x1 < x + croped_width and x2 > x and y1 < y + height and y2 > y:
                    x_left = x1 if x1 >= x else x
                    x_right = x2 if x2 <= x + croped_width else x + croped_width
                    y_up = y1 if y1 >= y else y
                    y_down = y2 if y2 <= y + height else y+ height
                else:
                    continue
                
                w = x_right - x_left
                h = y_down - y_up
                
                if (x_right - x_left) >= box[2]*0.7 and (y_down - y_up) >= box[3]*0.7:
                    #物件的70%包含在圖片裡
                    rescale_box = []
                    rescale_box.append((x_left + x_right)/2-x)
                    rescale_box.append((y_up + y_down)/2-y)
                    rescale_box.append(w)
                    rescale_box.append(h)
                    boxes_list[i].append(rescale_box)

        print(f'Sliding windows...Saved Images at {savePath}, labels at {cropAnno}')
        for num,img in tqdm.tqdm(enumerate(img_list)):
            saveImgPath = savePath + '/' + '{}_{}.jpg'.format(name, num)
            cv2.imwrite(saveImgPath, img)
            
            with codecs.open(cropAnno + '/' + '{}_{}.txt'.format(name, num) ,'w',encoding='utf-8') as f:
                boxes = boxes_list[num]
                if len(boxes) == 0:
                    continue
                for box in boxes:
                    f.write('0 '+str(box[0]/croped_width)+' '+str(box[1]/croped_height)+' '+str(box[2]/croped_width)+' '+str(box[3]/croped_height)+'\n')


if __name__ == '__main__':
    scale = 1
    windows_width = 3840 
    windows_height = 1561 
    imgpath = './fusion_image/test_itri/images_whole'
    srcAnn = './fusion_image/test_itri/labels_whole'
    annotation = './fusion_image/test_itri/test.txt'
    cropAnno = './fusion_image/test_itri/labels_test'
    savePath = './fusion_image/test_itri/images_test'
    
    crop_dataset(imgpath, windows_width, windows_height, srcAnn,  annotation, cropAnno, savePath)
