from sklearn.preprocessing import LabelEncoder
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
    # Kiểm tra hình ảnh không rỗng trước khi thay đổi kích thước
    if image is None or image.size == 0:
        raise ValueError("Empty image detected")
    return cv2.resize(image, size).flatten()

image_files = [f for f in glob.glob(os.path.join(dataset_path + "/**/*"), recursive=True) if not os.path.isdir(f)] 
random.seed(42)
random.shuffle(image_files)

for (i, img) in enumerate(image_files):
    image = cv2.imread(img)
    
    # Thêm kiểm tra rỗng
    if image is None:
        print(f"Warning: Could not read image {img}")
        continue
    
    pixels = image_to_feature_vector(image)
    data.append(pixels)

    label = img.split(os.path.sep)[-2].split(".")[0]
    labels.append(label)
    if i > 0 and i % 10 == 0:
        print("[INFO] processed {}/{}".format(i, len(image_files))) 

data = np.array(data)
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5, n_jobs=2)
model.fit(trainX, trainY)
acc = model.score(testX, testY)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

predictions = model.predict(testX)
print(classification_report(testY, predictions))

# Đọc lại và kiểm tra hình ảnh mới 
# im_path = "3.jpg"
im_path  = os.path.join(base_path, "3.jpg")
im = cv2.imread(im_path)

# Thêm kiểm tra để đảm bảo rằng hình ảnh được đọc đúng
if im is None:
    raise ValueError(f"Could not read image {im_path}")

im = image_to_feature_vector(im)
im = im.reshape(1, -1)

y = model.predict(im)
print(y)