
import numpy as np
import cv2 as cv
import sys


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_id = input('\n enter user id end press <return> ==>  ')
face_id = sys.argv[1]
if not cap.isOpened():
    #print("Cannot open camera")
    exit()
count =0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x,y,w,h) in faces:

        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv.putText(frame, 'photo count' + str(count), (x + 50, y + w + 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Save the captured image into the datasets folder
        cv.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv.imshow('saved_frame',  gray[y:y+h,x:x+w])
        print('click '+ str(count)+' photo')
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    elif count >= 30: # Take 30 face sample and stop video
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()