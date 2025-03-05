import os
import cv2
import numpy as np
import argparse
from   tqdm import tqdm
from   insightface_func.face_detect_crop_single import Face_detect
import glob

"""将视频crop为指定的框大小"""
def video_crop_specify_size(args):
    video_path      = config.video_path
    # crop size should based on the image resolution
    crop_size       = config.crop_size 
    video_basename  = os.path.splitext(os.path.basename(video_path))[0]
    output_path     = os.path.join(config.output_path, video_basename)
    os.makedirs(output_path, exist_ok=True)

    detect      = Face_detect(name='antelope', root='./insightface_func/models')
    detect.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    video       = cv2.VideoCapture(video_path)
    fps         = video.get(cv2.CAP_PROP_FPS)
    ret         = True
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            if frame_index == config.key_frame:
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
                    # print(f"up:{up}")
                    # print(f"left:{left}")
                    # print(f"bottom:{bottom}")
                    # print(f"right:{right}")
                    frame = cv2.rectangle(frame, (left,up), (right,bottom),(0,0,255),10)
                    cv2.imwrite(os.path.join(output_path, 'excample.JPG'.format(frame_index)), frame)
                    np.save(os.path.join(output_path, "location.npy"),(left,up,right,bottom))
                    break
        else:
            break
    video.release()
    video       = cv2.VideoCapture(video_path)
    ret         = True
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            frame  = frame[up:bottom,left:right,:]
            if config.output_size != config.crop_size:
                frame = cv2.resize(frame,(config.output_size, config.output_size),interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(output_path, '{:0>5d}.png'.format(frame_index)), frame)
        else:
            break
    video.release()
    return fps

def merge_video(args, fps):
    out_frame_dir   = args.output_path
    out_video_file  = args.video_out
    frame_path_list = sorted(glob.glob(out_frame_dir + "/*.png"))
    first_frame     = cv2.imread(frame_path_list[0])
    fourcc          = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    dest_fps        = fps
    height, width, _ = first_frame.shape
    video_writer    = cv2.VideoWriter(out_video_file, fourcc, dest_fps, (width, height))
    for frame in frame_path_list:
        img = cv2.imread(frame)
        video_writer.write(img)
    video_writer.release()
    print(f"The specify resolution of {args.output_size}x{args.output_size} is saved!!")

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, default='/home/ubuntu/liwen/video-crop-tool/1740625891727-src_audio.mp4',# 
                                                help="file path for input video") #
    parser.add_argument('-o', '--output_path', type=str, default='/home/ubuntu/liwen/video-crop-tool/crop_test/1740625891727-src_audio',# 
                                                help="file path to save edited video")
    parser.add_argument("-video_out", type=str, default='./output_demo.mp4', help="file path to save edited video") 
    parser.add_argument('--key_frame', type=int, default=0) # you need choose a frame as key frame,just roughtly define the face region
    parser.add_argument('-c', '--crop_size', type=int, default=300) # 660
    parser.add_argument('--output_size', type=int, default=512) #
    return parser.parse_args()


if __name__ == "__main__":
    config = getParameters()
    fps = video_crop_specify_size(args=config)
    merge_video(args=config, fps=fps)