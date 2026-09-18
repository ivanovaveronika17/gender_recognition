import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# 1. Streamlit поставувања
# --------------------------------------------------

st.set_page_config(
    page_title="GRDA",
    page_icon="👤",
    layout="centered"
)


# --------------------------------------------------
# 2. Наслов
# --------------------------------------------------

st.title("👤 Gender Recognition Demo App")

st.write(
    "Upload an image containing one or more faces "
    "and the application will predict the category "
    "for each detected face."
)


# --------------------------------------------------
# 3. Вчитување на face detector
# --------------------------------------------------

face_model_path = (
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

face_config_path = "models/deploy.prototxt"

face_net = cv2.dnn.readNetFromCaffe(
    face_config_path,
    face_model_path
)


# --------------------------------------------------
# 4. Вчитување на обучениот gender model
# --------------------------------------------------

gender_model = tf.keras.models.load_model(
    "models/gender_model.keras"
)


# --------------------------------------------------
# 5. Upload на слика
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 6. Ако е внесена слика
# --------------------------------------------------

if uploaded_file is not None:

    # Вчитување на сликата
    image = Image.open(uploaded_file)

    # PIL -> NumPy
    image = np.array(image)

    # RGB -> BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Димензии
    h, w = image.shape[:2]


    # --------------------------------------------------
    # 7. Прикажување на оригиналната слика
    # --------------------------------------------------

    st.subheader("Uploaded image")

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )


    # --------------------------------------------------
    # 8. Подготовка за face detector
    # --------------------------------------------------

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )


    # --------------------------------------------------
    # 9. Face detection
    # --------------------------------------------------

    face_net.setInput(blob)

    detections = face_net.forward()


    # Број на лица
    count = 0


    # --------------------------------------------------
    # 10. Ги проверуваме сите детекции
    # --------------------------------------------------

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]


        # Ги земаме само доволно сигурните детекции
        if confidence > 0.3:

            # --------------------------------------------------
            # 11. Координати на лицето
            # --------------------------------------------------

            box = detections[0, 0, i, 3:7] * [
                w,
                h,
                w,
                h
            ]

            startX, startY, endX, endY = box.astype("int")


            # --------------------------------------------------
            # 12. Padding околу лицето
            # --------------------------------------------------

            padding = 0.2

            face_width = endX - startX
            face_height = endY - startY

            startX = int(
                startX - face_width * padding
            )

            startY = int(
                startY - face_height * padding
            )

            endX = int(
                endX + face_width * padding
            )

            endY = int(
                endY + face_height * padding
            )


            # --------------------------------------------------
            # 13. Проверка на границите
            # --------------------------------------------------

            startX = max(0, startX)
            startY = max(0, startY)

            endX = min(w, endX)
            endY = min(h, endY)


            # --------------------------------------------------
            # 14. Crop на лицето
            # --------------------------------------------------

            face = image[
                startY:endY,
                startX:endX
            ]


            # Ако crop-от е празен
            if face.size == 0:
                continue


            # --------------------------------------------------
            # 15. BGR -> RGB
            # --------------------------------------------------

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )


            # --------------------------------------------------
            # 16. Resize
            # --------------------------------------------------

            face = cv2.resize(
                face,
                (224, 224)
            )


            # --------------------------------------------------
            # 17. Го зголемуваме бројот на лица
            # --------------------------------------------------

            count += 1


            # --------------------------------------------------
            # 18. Prediction
            # --------------------------------------------------

            processed_face = face / 255.0

            processed_face = np.expand_dims(
                processed_face,
                axis=0
            )

            prediction = gender_model.predict(
                processed_face,
                verbose=0
            )

            score = prediction[0][0]


            # --------------------------------------------------
            # 19. Одредување категорија
            # --------------------------------------------------

            if score >= 0.5:

                label = "Female"
                gender_score = score

            else:

                label = "Male"
                gender_score = 1 - score


            # --------------------------------------------------
            # 20. Прикажување резултат
            # --------------------------------------------------

            st.subheader(
                f"Face {count}"
            )

            col1, col2 = st.columns(2)


            # Crop
            with col1:

                st.image(
                    face,
                    caption="Detected face",
                    width=180
                )


            # Prediction
            with col2:

                st.write(
                    f"**Prediction:** {label}"
                )

                st.write(
                    f"**Score:** "
                    f"{gender_score * 100:.1f}%"
                )


                if label == "Female":

                    st.success(
                        f"Female — "
                        f"{gender_score * 100:.1f}%"
                    )

                else:

                    st.info(
                        f"Male — "
                        f"{gender_score * 100:.1f}%"
                    )


            st.divider()


    # --------------------------------------------------
    # 21. Ако нема детектирани лица
    # --------------------------------------------------

    if count == 0:

        st.warning(
            "No faces were detected in the image."
        )

    else:

        st.success(
            f"Detected faces: {count}"
        )