import cv2
from pathlib import Path

from preprocessing import preprocess_face


# Го вчитуваме DNN face detector моделот
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_path, model_path)


# Папки
image_folder = Path("test_images")
output_folder = Path("cropped_faces")

# Ако папката не постои, ја креираме
output_folder.mkdir(exist_ok=True)


# Ги поминуваме сите слики
for image_path in image_folder.iterdir():

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Не можев да ја прочитам: {image_path.name}")
        continue

    h, w = image.shape[:2]

    # Ја подготвуваме сликата за DNN моделот
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)

    # Ги добиваме детекциите
    detections = net.forward()

    face_number = 0

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Го прифаќаме лицето ако confidence е над 30%
        if confidence > 0.3:

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            startX, startY, endX, endY = box.astype("int")

            # Ги ограничуваме координатите да останат во сликата
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Го сечеме само лицето
            face = image[startY:endY, startX:endX]

            if face.size == 0:
                continue

            # Го правиме preprocessing-от
            processed_face = preprocess_face(face)

            # Го враќаме од 0-1 назад во 0-255
            processed_face = (processed_face * 255).astype("uint8")

            face_number += 1

            # Име на излезниот фајл
            output_path = output_folder / f"{image_path.stem}_face{face_number}.jpg"

            # Ја зачувуваме исечената слика
            cv2.imwrite(str(output_path), processed_face)

            print(f"Зачувано: {output_path}")