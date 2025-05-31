import cv2 as cv
import os

# Load the folder where the face data is stored
data_path = 'path folder where the face data is stored'
data_list = os.listdir(data_path)

# Load the trained face recognition model
face_recognizer = cv.face.EigenFaceRecognizer_create()
face_recognizer.read('EigenFaceRecognizerTraining.xml')

# Load face detection classifier
face_cascade = cv.CascadeClassifier('load face detection classifies/OpencvRuidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

# Activate the camera
camera = cv.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_copy = gray.copy()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        captured_face = gray_copy[y:y+h, x:x+w]
        captured_face = cv.resize(captured_face, (160, 160), interpolation=cv.INTER_CUBIC)
        result = face_recognizer.predict(captured_face)
        cv.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (0, 255, 0), 1, cv.LINE_AA)

        if result[1] < 8000:
            cv.putText(frame, '{}'.format(data_list[result[0]]), (x, y - 20), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv.putText(frame, "Not recognized", (x, y - 20), 2, 0.7, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow("Results", frame)
    if cv.waitKey(1) == ord('s'):
        break

camera.release()
cv.destroyAllWindows()
