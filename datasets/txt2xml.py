import os
import cv2

def convert_txt_to_xml(txt_file, xml_file, img_file):
    """
    將 .txt 檔案轉換成 .xml 檔案

    Args:
        txt_file: .txt 檔案路徑
        xml_file: .xml 檔案路徑
        img_file: .jpg 檔案路徑
    """

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 解析 .txt 檔案
    image_name = os.path.basename(txt_file)[:-4]
    img = cv2.imread(img_file)
    height, width, depth=img.shape

    # 建立 .xml 檔案

    with open(xml_file, 'w') as f:
        f.write('<annotation>\n')
        f.write('  <folder>images</folder>\n')
        f.write('  <filename>{}</filename>\n'.format(image_name))
        f.write('  <size>\n')
        f.write('    <width>{}</width>\n'.format(width))
        f.write('    <height>{}</height>\n'.format(height))
        f.write('    <depth>{}</depth>\n'.format(depth))
        f.write('  </size>\n')
        f.write('  <segmented>3</segmented>\n')

        for line in lines:
            label, x, y, w, h = line.split(' ')
            xmin = int((float(x) - float(w)/2) * width)
            ymin = int((float(y) - float(h)/2) * height)
            xmax = int((float(x) + float(w)/2) * width)
            ymax = int((float(y) + float(h)/2) * height)

            f.write('  <object>\n')
            f.write('    <name>{}</name>\n'.format('crater'))
            f.write('    <pose>Unspecified</pose>\n')
            f.write('    <truncated>0</truncated>\n')
            f.write('    <difficult>0</difficult>\n')
            f.write('    <bndbox>\n')
            f.write('      <xmin>{}</xmin>\n'.format(xmin))
            f.write('      <ymin>{}</ymin>\n'.format(ymin))
            f.write('      <xmax>{}</xmax>\n'.format(xmax))
            f.write('      <ymax>{}</ymax>\n'.format(ymax))
            f.write('    </bndbox>\n')
            f.write('  </object>\n')

        f.write('</annotation>\n')


if __name__ == '__main__':
    # 設定 .txt 檔案和 .xml 檔案的儲存路徑
    txt_dir = './fusion_image/test_google/labels/'
    img_dir = './fusion_image/test_google/images/'
    xml_dir = './fusion_image/test_google/label_xml/'

    # 批量轉換檔案
    for file in os.listdir(txt_dir):
        txt_file = os.path.join(txt_dir, file)
        xml_file = os.path.join(xml_dir, file[:-4] + '.xml')
        img_file = os.path.join(img_dir, file[:-4] + '.jpg')
        convert_txt_to_xml(txt_file, xml_file, img_file)
