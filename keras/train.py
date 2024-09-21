import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model.vgg import vgg
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", type=str, default="gender_detection.model",
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = ap.parse_args()

# Cấu hình huấn luyện
epochs = 100  # Tăng số lượng epochs
lr = 1e-4    # Giảm learning rate
batch_size = 64
img_dims = (48, 48, 3)

data = []
labels = []

# Xây dựng đường dẫn tuyệt đối cho thư mục dataset
base_path = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(base_path, args.dataset)
model_path = os.path.join(base_path, args.model)
if not os.path.exists(model_path):
    print("123")
print(model_path)
plot_path = os.path.join(base_path, args.plot)
# Tìm tất cả các tệp ảnh trong thư mục
image_files = [f for f in glob.glob(os.path.join(dataset_path + "/**/*"), recursive=True) if not os.path.isdir(f)]
random.seed(42)
random.shuffle(image_files)

# Xử lý ảnh
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  # Lấy tên nhãn từ tên thư mục
    if label == "nu":
        label = 1
    else:
        label = 0
    labels.append([label])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Kỹ thuật tăng cường dữ liệu
aug = ImageDataGenerator(rotation_range=40,  # Tăng giá trị của rotation_range
                         width_shift_range=0.2,  # Tăng tỷ lệ dịch chuyển chiều rộng
                         height_shift_range=0.2,  # Tăng tỷ lệ dịch chuyển chiều cao
                         shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Xây dựng mô hình
model = vgg.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Sử dụng optimizer của legacy
opt = tf.keras.optimizers.legacy.Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Huấn luyện mô hình
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX,testY), steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# Lưu mô hình
model.save(model_path)

# Vẽ đồ thị quá trình huấn luyện
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(plot_path)

# Đánh giá mô hình
scores = model.evaluate(testX, testY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))