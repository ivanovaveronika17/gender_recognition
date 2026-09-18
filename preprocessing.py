import cv2


def preprocess_face(face):
    # Ја менуваме големината на лицето на 224 x 224
    face = cv2.resize(face, (224, 224))

    # Ги претвораме вредностите на пикселите од 0-255 во 0-1
    face = face / 255.0

    return face