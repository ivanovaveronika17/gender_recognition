import cv2
from pathlib import Path

model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(config_path, model_path)

image_folder = Path("test_images")

for image_path in image_folder.iterdir():

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Не можев да ја прочитам сликата: {image_path.name}")
        continue

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)

    detections = net.forward()

    count = 0

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(
                image,
                (startX, startY),
                (endX, endY),
                (255, 0, 0),
                2
            )

            text = f"{confidence * 100:.1f}%"

            cv2.putText(
                image,
                text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

            count += 1

    print(f"{image_path.name}: пронајдени лица = {count}")

    cv2.imshow(image_path.name, image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
