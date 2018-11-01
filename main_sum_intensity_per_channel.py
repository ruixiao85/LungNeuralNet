import argparse
from osio import find_file_recursive
import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and predict with biomedical images.')
    parser.add_argument('-d', '--dir', dest='dir', action='store',
                        default='D:/Cel files/2018-10.25 Uchenna Green Blue', help='work directory')
    parser.add_argument('-e', '--ext', dest='ext', action='store',
                        default='*.tif', help='extension')
    args = parser.parse_args()

    print("filename,blue_sum,green_sum,red_sum")
    for img in find_file_recursive(args.dir,args.ext):
        im=cv2.imread(img)
        # print(im.dtype)
        # print(im.shape)
        print(img+','+np.array2string(np.sum(im,axis=(0,1)),separator=','))
        # print(img+','+','.join(str(n) for n in np.sum(im,axis=(0,1)).tolist()))
