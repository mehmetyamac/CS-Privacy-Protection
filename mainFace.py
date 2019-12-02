### Face Recognition Attack of Type-I and Type-II ###
### Dlib's pretrained networks:
'''You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
'''
import os
import copy
import pickle
from imutils import paths
import numpy as np

import faceRec # Utils for face recognition.

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Output path of Privacy Presenvation MATLAB code.
data_dir = 'C:\\Users\\ahishalm\\Desktop\\projects\\pri_pre\\Results\\measurement_6'

# Calculate the face recognition over User-A or User-B or original ('userA_faces', 'userB_faces', 'original_faces').
user = 'userB_faces'
atackType = 'AttackI' # Attack type: 'AttackI' or 'AttackII'

# Original train data, note that this contains the same samples for all measurement rates or degradation methods unless it is not AttackII !!!
train_dir = data_dir + '\\train_faces' + atackType # Original train data

# Read face images, extract face features and identity names, finally write the computed features
if atackType is 'AttackI': # Check if there is already computed face embeddings for AttackI, since they are the same for this particular attack.
    if os.path.exists('face_embeddingsAttackI.pickle') == False: faceRec.write_embeddings(train_dir, atackType)
else:
    faceRec.write_embeddings(train_dir, atackType)

test_dir = data_dir + '\\' + user
data = pickle.loads(open('face_embeddings' + atackType + '.pickle', "rb").read()) # Load face embedings.
embeddings = np.array(data["embeddings"])
names = np.array(data["names"])

faces = list(paths.list_images(test_dir))

total = 0
true = 0

# Face recognition over the test samples:
for (i, fileName) in enumerate(faces):
    name = fileName.split(os.path.sep)[-2] # The true identity name from the image folder.
    total += 1
    true += faceRec.recognize_image(name, embeddings, names, fileName) # Perform face recognition and return 1 if the estimation is correct.
    
print('Recognized True: ', true)
print('Total number of frames: ', total)
print('Accuracy: ', true/total)
