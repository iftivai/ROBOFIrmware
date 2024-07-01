#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'
                                   )

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

while True:
    (sucess, imgOrignal) = cap.read()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for (x, y, w, h) in faces:
        image = imgOrignal[y:y + h, x:x + h]
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = image_array.astype(np.float32) / 127.0 \
            - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        classIndex = prediction.argmax(axis=-1)
        print(classIndex)
    cv2.imshow('Result', imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
