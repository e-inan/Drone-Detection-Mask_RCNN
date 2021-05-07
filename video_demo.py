import cv2
from visualize_cv2 import model, display_instances, class_names
import sys

# args = sys.argvqq
# if(len(args) < 2):
# 	print("run command: python video_demo.py 0 or video file name")
# 	sys.exit(0)
# name = args[1]
# if(len(args[1]) == 1):
# 	name = int(args[1])
	
scale_percent = 75/ 100 # percent of original size

name = "/home/buzun/Workspace/Mask_RCNN/classroom.mp4"
stream = cv2.VideoCapture(name)
	
while True:
    ret , frame = stream.read()
    if not ret:
        print("unable to fetch frame")
        break
    
    # # resize image
    # width = int(frame.shape[1] * scale_percent)
    # height = int(frame.shape[0] * scale_percent)
    # frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
    
    results = model.detect([frame], verbose=0)

    # Visualize results
    r = results[0]
    masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.imshow("masked_image",masked_image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

stream.release()
cv2.destroyAllWindows()