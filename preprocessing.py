import cv2


def preprocess_face(face):
   
    face = cv2.resize(face, (224, 224))

  
    face = face / 255.0

    return face
