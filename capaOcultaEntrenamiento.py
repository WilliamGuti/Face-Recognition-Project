import cv2 
import numpy as np
import os
from time import time

# load the folder where the face data is stored
data_path = 'path where you'll charge data capture'
data_list = os.listdir(data_path)

# lists to store image data and corresponding labels
labels = []
face_data = []
label_id = 0

start_time = time()

for folder_name in data_list:
    full_path = data_path + '/' + folder_name
    print('Starting to read images...')
    
    for file in os.listdir(full_path):
        print('Image:', folder_name + '/' + file)
        labels.append(label_id)
        # Load image in grayscale
        face_data.append(cv2.imread(full_path + '/' + file, 0)) 
        
    label_id += 1

    read_end_time = time()
    read_duration = read_end_time - start_time
    print('Total Reading Time:', read_duration)

# training the face recognition model
face_recognizer = cv2.face.EigenFaceRecognizer_create()
print("Starting training... Please wait.")
face_recognizer.train(face_data, np.array(labels))
training_end_time = time()
training_duration = training_end_time - start_time
print("Total Training Time:", training_duration)

face_recognizer.write('EigenFaceRecognizerTraining.xml')
print('Training Completed')
