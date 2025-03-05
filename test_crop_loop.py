import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
import time


face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_resnet_facedetection_scrfd10gkps')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_detection2.jpeg'
result = face_detection(img_path)

# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
s = time.time()
img_draw = draw_face_detection_result('srcImg.jpg', result)
e = time.time()
print(f"{e - s} s")