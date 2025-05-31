import cv2 as cv
import numpy as np
import os  # used to create directories with Python code
import imutils

model_name = 'WilliamPhotos'  # name of the folder to be created
base_path = 'C:/Users/w/OneDrive/Desktop/proyectoReconocimientoDeImagenes/reconocimientoFacial1CapturaDatos'
full_path = base_path + '/' + model_name

# create the directory if it doesn't exist
if not os.path.exists(full_path):
    os.makedirs(full_path)

# pre-trained classifier for frontal face detection
face_cascade = cv.CascadeClassifier(
    'C:/Users/w/OneDrive/Desktop/proyectoReconocimientoDeImagenes/entrenamientosOpencvRuidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml'
)

# activate webcam
camera = cv.VideoCapture(0)

# image ID
img_id = 0

while True:
    success, frame = camera.read()
    if not success:
        print("Cannot access the camera")
        break

    # resize frame in case of high resolution
    frame = imutils.resize(frame, width=640)
    # convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)

    # make a copy of the original frame for cropping
    frame_copy = frame.copy()

    # draw rectangles around detected faces and save them
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # crop the face from the frame
        captured_face = frame_copy[y:y + h, x:x + w]
        # resize the cropped face
        captured_face = cv.resize(captured_face, (160, 160), interpolation=cv.INTER_CUBIC)
        # save the image to the folder
        cv.imwrite(full_path + '/image_{}.jpg'.format(img_id), captured_face)
        # increment the image ID
        img_id += 1

    # show the frame with rectangles
    cv.imshow("Face Detection Result", frame)

    # stop when 351 images have been saved
    if img_id == 351:
        break

camera.release()
cv.destroyAllWindows()
