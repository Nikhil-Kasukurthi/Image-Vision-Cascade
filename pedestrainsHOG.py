import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.ocl.setUseOpenCL(True)

video = cv2.VideoCapture('./Urban/march9.avi')

while True:
	ret, image = video.read()
	#image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	image = imutils.resize(image, width=min(600, image.shape[1]))
	orig = image.copy()
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(4, 4), scale=1.05)
 
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.2)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	# filename = imagePath[imagePath.rfind("/") + 1:]
	# print("[INFO] {}: {} original boxes, {} after suppression".format(
	# 	filename, len(rects), len(pick)))
 

	cv2.imshow('Videa',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
