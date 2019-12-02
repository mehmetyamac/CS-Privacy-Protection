import os
import pickle
from imutils import paths
import numpy as np

import dlib

sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat') # Dlib's landmark detector.
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat') # Dlib's pretrained resnet face feature extractor.

def write_embeddings(data_dir, atackType):

    faces = list(paths.list_images(data_dir)) # Train samples directory for query.
    embeddings = []
    names = []

    for (i, fileName) in enumerate(faces): # Compute face embeddings for the trainset.
        name = fileName.split(os.path.sep)[-2] # The person's name.
        image = dlib.load_rgb_image(fileName)
        d = dlib.rectangle(left = 0, top = 0, right = image.shape[0], bottom = image.shape[1]) # Sample is already in cropped face form.
        shape = sp(image, d) # Landmark detector.
        face_descriptor = facerec.compute_face_descriptor(image, shape) # Extracted face embeddings.
        
        embeddings.append(face_descriptor)
        names.append(name)
        data = {"embeddings": embeddings, "names": names}

    f = open('face_embeddings' + atackType + '.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Face embeddings are extracted and saved.")


def recognize_image(target, embeddings, names, fileName):
    
    image = dlib.load_rgb_image(fileName)
    tolerance = 0.6 # Distance tolerance for the nearest-search.
    d = dlib.rectangle(left = 0, top = 0, right = image.shape[0], bottom = image.shape[1]) # Sample is already in cropped face form.

    shape = sp(image, d) # Landmark detector.
    face_descriptor = facerec.compute_face_descriptor(image, shape) # Extracted face embeddings.

    ### Nearest-neighbor search: ###
    neighbors = np.linalg.norm(embeddings - face_descriptor, axis=1) <= tolerance
    names = names[neighbors]
    name = []
    IDcount = {}
    for i in range(0, len(names)): IDcount[names[i]] = IDcount.get(names[i], 0) + 1 # Check the identites of the found neighbors.

    if len(IDcount) > 0 : name = max(IDcount, key = IDcount.get) # Nearest identity.

    flag = 0
    if name == target: # Correctly clasified.
        flag = 1
    return flag
