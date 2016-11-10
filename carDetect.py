import cv2
import sys

carCascade = cv2.CascadeClassifier('cars.xml')
pedestrianCascade = cv2.CascadeClassifier('pedestrian.xml')

video = cv2.VideoCapture('./Test/Urban/march9.avi')
#video = cv2.VideoCapture('./Test/Sunny/april21.avi')
#video = cv2.VideoCapture('./Test/Dense/jan28.avi')

while True:
	ret, image = video.read()
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	cars = carCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=10
		)
	# cars = carCascade.detectMultiScale(
	#     gray,
	#     scaleFactor=2,
	#     minNeighbors=10
	# )
	pedestrians = pedestrianCascade.detectMultiScale(
		gray,
		scaleFactor = 3,
		minNeighbors = 10
		)

#	Draw a rectangle around the cars
	for (x, y, w, h) in cars:
	    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
	
	for (x, y, w, h) in pedestrians:
	    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
	
	cv2.imshow('Video',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture

cv2.destroyAllWindows()