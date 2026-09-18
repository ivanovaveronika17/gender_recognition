import cv2
from pathlib import Path

from preprocessing import preprocess_face


model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_path, model_path)



image_folder = Path("test_images")
output_folder = Path("cropped_faces")

output_folder.mkdir(exist_ok=True)


for image_path in image_folder.iterdir():

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Не можев да ја прочитам: {image_path.name}")
        continue

    h, w = image.shape[:2]


    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)

    detections = net.forward()

    face_number = 0

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            startX, startY, endX, endY = box.astype("int")

            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

           
            face = image[startY:endY, startX:endX]

            if face.size == 0:
                continue

           
            processed_face = preprocess_face(face)

           
            processed_face = (processed_face * 255).astype("uint8")

            face_number += 1

           
            output_path = output_folder / f"{image_path.stem}_face{face_number}.jpg"

           
            cv2.imwrite(str(output_path), processed_face)

            print(f"Зачувано: {output_path}")
