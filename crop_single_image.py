import os
import cv2
import numpy as np
import argparse
from   tqdm import tqdm
from   insightface_func.face_detect_crop_single import Face_detect
import glob

def video_crop_specify_size(args):
    image_dir      = config.image_dir
    frame_list = glob.glob(image_dir + "/*.png")
    print(frame_list)
    for frame_path in sorted(frame_list):
        image_basename = os.path.basename(os.path.splitext(frame_path)[0])
        # crop size should based on the image resolution
        crop_size       = config.crop_size 
        os.makedirs(config.output_path, exist_ok=True)
        output_cropped  = os.path.join(config.output_path, image_basename + ".png")
        print(output_cropped)

        detect      = Face_detect(name='antelope', root='./insightface_func/models')
        detect.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
        frame = cv2.imread(frame_path)
        detect_results, bbox_list  = detect.get_kpts(frame)

        if detect_results:
            shape  = frame.shape
            kpts   = detect_results[0][2] # 鼻尖
            print("Find keypoint coodination:", kpts)
            left   = kpts[0]- crop_size//2  if kpts[0] - crop_size//2 >=0 else 0
            temp   = crop_size//2 + crop_size//16
            up     = (kpts[1]-  temp)   if (kpts[1] - temp) >=0 else 0
            bottom = up + crop_size
            if bottom > shape[0]:
                bottom = shape[0]
                up     = bottom - crop_size
            right = left + crop_size
            if right > shape[1]:
                right = shape[1]
                left  = right - crop_size

            # Ensure the coordinates are within the frame boundaries
            up = max(0, min(up, shape[0] ))
            left = max(0, min(left, shape[1]))
            bottom = max(0, min(bottom, shape[0]))
            right = max(0, min(right, shape[1]))

            up, left, bottom, right = int(up), int(left), int(bottom), int(right)
            # print(up, left, bottom, right ) # 179 1474 979 2274
            frame_cropped = frame[up:bottom, left:right,:]
            # print(f"up:{up}")
            # print(f"left:{left}")
            # print(f"bottom:{bottom}")
            # print(f"right:{right}")
            # frame = cv2.rectangle(frame, (left,up), (right,bottom),(0,0,255),10)

            cv2.imwrite(output_cropped, frame_cropped)

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--image_dir', type=str, default='/home/ubuntu/liwen/video-crop-tool/avata_frame',# 
                                                help="file path for input video") #
    parser.add_argument('-o', '--output_path', type=str, default='/home/ubuntu/liwen/video-crop-tool/test_avartar',# 
                                                help="file path to save edited video")
    parser.add_argument('-c', '--crop_size', type=int, default=800) # 660
    parser.add_argument('--output_size', type=int, default=512) #
    return parser.parse_args()


if __name__ == "__main__":
    config = getParameters()
    video_crop_specify_size(args=config)