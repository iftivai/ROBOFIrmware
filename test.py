import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
from keras.models import load_model
from PIL import Image, ImageOps


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX


model = load_model('keras_model.h5')


def get_className(classNo):
	if classNo==0:
		return "Chando"
	elif classNo==1:
		return "Tony Stark"

while True:
	sucess, imgOrignal=cap.read()
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	image = Image.open('images/antora/31.jpg')
	
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (224,224))
		img=img.reshape(1, 224, 224, 3)
		prediction=model.predict(img)
		#classIndex=model.predict_classes(img)
		#classIndex = prediction.argmax(axis=-1)
		probabilityValue=np.amax(prediction)

		size = (224, 224)
		image = ImageOps.fit(img, size, Image.ANTIALIAS)

#turn the image into a numpy array
		image_array = np.asarray(img)
# Normalize the image
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
		data[0] = normalized_image_array

# run the inference
		prediction = model.predict(data)
		classIndex = prediction.argmax(axis=-1)
		print(prediction)
		if classIndex==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		elif classIndex==1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()





















