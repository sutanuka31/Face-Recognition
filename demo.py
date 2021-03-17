# All the imports go here
import numpy as np
from cv2 import cv2
# import sys
from os import path

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   #load trained model
id = 2  #two persons
names = ['','Arpita','sutanuka','unknown']  #key in names, start from the second place, leave first empty

font = cv2.FONT_HERSHEY_SIMPLEX
face_id = input('\n enter user id end press <return> ==>  ')

# face_id = sys.argv[1]
# Variable store execution state
first_read = True

# Starting the video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, img = cap.read()
count =0
while (ret):
    ret, img = cap.read()
    # Coverting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    if (len(faces) > 0):

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # count += 1
            # roi_face is face which is input to eye classifier
            # cv2.putText(img, 'photo count' + str(count), (x + 50, y + w + 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            roi_face = gray[y:y + h, x:x + w]
            # roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))


            # Examining the length of eyes object for eyes
            if (len(eyes) >= 2):

                # Check if program is running for detection
                if (first_read):
                    cv2.putText(img,
                                "Eye detected",
                                (70, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 255, 0), 2)

                    count += 1
                    # if count <= 1:
                    if (confidence < 100):
                        id = names[id]
                        confidence = "{0}%".format(round(100 - confidence))
                        # cv2.putText(img,
                        #     "Eyes open!", (70, 70),
                        #      cv2.FONT_HERSHEY_PLAIN, 2,
                        #      (255, 255, 255), 2)

                        # print('click ' + str(count) + ' photo',confidence,id)
                        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
                        #cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                        if (confidence > "30%"):
                            cv2.putText(img, 'Already in the dataset', (x + 50, y + w + 20), font, 1, (255, 255, 0), 2)
                        # cv2.imwrite("dataset/User." + str(face_id) + '.' +str(count)+ ".jpg", gray[y:y + h, x:x + w])
                        else:
                            cv2.putText(img, 'New face was detected', (x + 50, y + w + 20), font, 1,
                                        (255, 255, 0), 1)
                            print('click ' + str(count) + ' photo',confidence,id)
                            # cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) +id+ ".jpg",
                            #             gray[y:y + h, x:x + w])
                    first_read = True
                    # else:
                    #     pass
            else:
                if (first_read):
                    # To ensure if the eyes are present before starting
                    cv2.putText(img,
                                "No eyes detected", (70, 70),
                                cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 2)
                else:
                    # This will print on console and restart the algorithm
                    # cv2.imwrite("dataset/User." + str(face_id) + '.' +str(count)+ ".jpg", gray[y:y + h, x:x + w])

                    cv2.waitKey(30)
                    first_read = True
    else:
        cv2.putText(img,
                    "No face detected", (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 2)
        # Controlling the algorithm with keys
    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if (a == ord('q')):
        break
    elif (a == ord('s') and first_read):
        # This will start the detection
        first_read = False
    elif count >= 30:  # Take 30 face sample and stop video
        break


cap.release()
cv2.destroyAllWindows()