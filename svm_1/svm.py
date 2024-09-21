
import matplotlib
matplotlib.use("Agg")
from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from imutils import paths
import imutils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob
from sklearn.neighbors import KNeighborsClassifier

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
	help="path to input dataset")
args = ap.parse_args()
base_path = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(base_path, args.dataset)
data = []
labels = []
def image_to_feature_vector(image, size=(32, 32)):
  
    return cv2.resize(image, size).flatten()

image_files = [f for f in glob.glob(os.path.join(dataset_path + "/**/*"), recursive=True) if not os.path.isdir(f)] 
random.seed(42)
random.shuffle(image_files)

for (i,img) in enumerate(image_files):
    
    image = cv2.imread(img)
    
    pixels = image_to_feature_vector(image)
    
    data.append(pixels)

    label = img.split(os.path.sep)[-2].split(".")[0]
    labels.append(label)
    if i > 0 and i % 600 == 0:
        print("[info] processed {}/{}".format(i , len(image_files))) 
	
(trainX, testX, trainY, testY) = train_test_split(np.array(data), labels, test_size=0.2,
                                                  random_state=42)


model = LinearSVC()

model.fit(trainX, trainY)
acc = model.score(testX, testY)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

predictions = model.predict(testX)
print(classification_report(testY, predictions
	))
im_path  = os.path.join(base_path, "22.jpg")
im = cv2.imread(im_path)
# im = cv2.imread("22.jpg")
im = image_to_feature_vector(im)
   
im = im.reshape(1, -1)

y = model.predict(im)

print(y)



