
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = ap.parse_args()


# dwnld_link = "pre-trained-weights/gender_detection.model"
# model_path = get_file("gender_detection.model", dwnld_link,
#                      cache_subdir="pre-trained", cache_dir=os.getcwd())
base_path = os.path.abspath(os.path.dirname(__file__))

# Đường dẫn tuyệt đối đến mô hình đã lưu
model_path = os.path.join(base_path, "gender_detection.model")         

model = load_model(model_path)
im_path  = os.path.join(base_path, "anhtest/3.jpg")
# im = cv2.imread(im_path)
# im = cv2.imread("bg.jpg")

image = cv2.imread(im_path)

if image is None:
    print("Could not read input image")
    exit()


model = load_model(model_path)


face, confidence = cv.detect_face(image)

classes = ['nam','nu']


for i, f in enumerate(face):

           
    (sX, sY) = f[0], f[1]
    (endX, endY) = f[2], f[3]

 
    cv2.rectangle(image, (sX,sY), (endX,endY), (0,255,0), 2)

  
    cat = np.copy(image[sY:endY,sX:endX])
    
    
    cat = cv2.resize(cat, (48,48))
    
    cat = cat.astype("float") / 255.0
    
    cat = img_to_array(cat)
    cat = np.expand_dims(cat, axis=0)
    
    
    d = model.predict(cat)[0]
   

  
    i = np.argmax(d)
    label = classes[i]

    label = "{}: {:.2f}%".format(label, d[i] * 100)

    Y = sY - 10 if sY - 10 > 10 else sY + 10

   
    cv2.putText(image, label, (sX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)


cv2.imshow("gender detection", image)

          
cv2.waitKey()


cv2.imwrite("gender_detection.jpg", image)


cv2.destroyAllWindows()
