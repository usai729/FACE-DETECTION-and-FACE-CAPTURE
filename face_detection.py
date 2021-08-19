import cv2 as cv
import glob
import numpy as np

# Trained face detection data
face_detection_data = cv.CascadeClassifier('face_detection_data.xml')

img = cv.VideoCapture(0)    #Capture your own image
count = 1
imagesCaptured = []

while True:
    isTrue, frame = img.read()

    if isTrue:
        grayScale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_detection_data.detectMultiScale(
            grayScale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if (len(faces) > 0):
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h),
                             (0, 150, 0), thickness=2)

                cv.imwrite('./Images/image_'+str(count) +
                           '.jpg', frame[y: y+h, x: x+w])
                count += 1

            cv.putText(frame, f"{len(faces)} face(s) detected", (10, 30),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 150, 255), thickness=2)
            cv.imshow("Detected Output", frame)
        else:
            cv.putText(frame, "No faces Detected!", (10, 30),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=2)

            cv.imshow("Detected Output", frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    else:
        break

img.release()
cv.destroyAllWindows()
