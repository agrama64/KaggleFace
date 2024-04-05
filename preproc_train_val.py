# import the necessary packages
from PIL import Image
import numpy as np
import cv2
import os
from time import sleep
from tqdm import tqdm
import random


#folder for processed images
if not(os.path.exists("train_proc") and os.path.isdir("train_proc")):
    os.mkdir("train_proc")
if not(os.path.exists("val_proc") and os.path.isdir("val_proc")):
    os.mkdir("val_proc")
# load our serialized model from disk


# This code and the following DNN is from: https://github.com/gopinath-balu/computer_vision/tree/master/CAFFE_DNN
prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
confidence_const = 0.5
imgdir = "/train/"

print(imgdir)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
#image = cv2.imread("train_small/100.jpg")
misses = 0


for j in tqdm(range(69540)): #remove a 0 for small dataset
	path = "train/" + str(j) + ".jpg"

	image_raw = Image.open(path).convert('RGB')
	image = np.array(image_raw)
	#image = np.asarray(image_raw, dtype=np.float32) / 255
	#image = image[:, :, :3]

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	#cv2.imshow("Output", blob)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	count = 0

	max_area = 0
	max_confidence = 0
	startX_fin = 0
	startY_fin = 0
	endX_fin = 0
	endY_fin = 0
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > confidence_const:
			count += 1
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			area_int = (abs(endX - startX)) * (abs(endY-startY))


			#if (area_int > max_area): #save bounds for biggest face detected
				#max_confidence = confidence
				#max_area = area_int
				#startX_fin = startX
				#startY_fin = startY
				#endX_fin = endX
				#endY_fin = endY

			if (confidence > max_confidence): #save bounds for highest-confidence face detected
				max_confidence = confidence
				max_area = area_int
				startX_fin = startX
				startY_fin = startY
				endX_fin = endX
				endY_fin = endY

	if (max_area != 0):
		rd = random.randint(1, 10)
		area = (startX_fin, startY_fin, endX_fin, endY_fin)
		cropped_img = image_raw.crop(area)
		path = "train_proc/" + str(j) + ".jpg"
		val_path = "val_proc/" + str(j) + ".jpg"

		if (rd < 4):
			cropped_img.save(val_path, 'JPEG')
		else:
			cropped_img.save(path, 'JPEG')
		#cropped_img.show()
	else:
		misses += 1


	#print('Count ', count)
	# show the output image
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)
		
print(misses)