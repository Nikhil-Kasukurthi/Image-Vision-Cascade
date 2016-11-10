import cv2

carCascade = cv2.CascadeClassifier('trafficSignsCascade.xml')

video = cv2.VideoCapture('./Dense/jan28.avi')

while True:
	ret, image = video.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cars = carCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=1
	)

	# Draw a rectangle around the cars
	for (x, y, w, h) in cars:
	    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
	cv2.imshow('Video',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture

cv2.destroyAllWindows()