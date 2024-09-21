from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Xác định đường dẫn tuyệt đối đến thư mục chứa tệp script
base_path = os.path.abspath(os.path.dirname(__file__))

# Đường dẫn tuyệt đối đến mô hình đã lưu
model_path = os.path.join(base_path, "gender_detection.model")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist.")


# Tải mô hình đã lưu
model = load_model(model_path)

# Khởi chạy webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# Danh sách các lớp (nhãn)
classes = ['nam', 'nu']

# Vòng lặp xử lý liên tục các khung ảnh từ webcam
while webcam.isOpened():
    # Đọc khung ảnh từ webcam
    status, frame = webcam.read()
    if not status:
        print("Could not read frame")
        exit()

    # Phát hiện khuôn mặt trong khung ảnh
    face, confidence = cv.detect_face(frame)
    for i, f in enumerate(face):
        (sX, sY) = f[0], f[1]
        (eX, eY) = f[2], f[3]

        # Vẽ hình chữ nhật bao quanh khuôn mặt
        cv2.rectangle(frame, (sX, sY), (eX, eY), (0, 255, 0), 2)

        # Trích xuất khuôn mặt từ khung ảnh
        cat = np.copy(frame[sY:eY, sX:eX])

        if (cat.shape[0]) < 10 or (cat.shape[1]) < 10:
            continue

        # Tiền xử lý khuôn mặt cho phù hợp với mô hình
        cat = cv2.resize(cat, (48, 48))
        cat = cat.astype("float") / 255.0
        cat = img_to_array(cat)
        cat = np.expand_dims(cat, axis=0)

        # Dự đoán giới tính
        d = model.predict(cat)[0]
        i = np.argmax(d)
        label = classes[i]

        # Định dạng nhãn và xác suất
        label = "{}: {:.2f}%".format(label, d[i] * 100)

        # Vị trí của nhãn
        Y = sY - 10 if sY - 10 > 10 else sY + 10

        # Vẽ nhãn lên khung ảnh
        cv2.putText(frame, label, (sX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị khung ảnh
    cv2.imshow("gender detection", frame)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và phá hủy các cửa sổ OpenCV
webcam.release()
cv2.destroyAllWindows()