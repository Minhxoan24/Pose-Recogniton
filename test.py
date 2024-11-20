import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skimage.feature import hog

# Hàm tiền xử lý ảnh, kết hợp Edge Detection và HOG từ skimage
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (128, 128))  # Resize ảnh về kích thước chuẩn
    
    # Edge Detection (Canny)
    edges = cv2.Canny(image_resized, 100, 200)  # Phát hiện biên
    edges_flattened = edges.flatten()  # Chuyển đổi thành vector một chiều

    # HOG (Histogram of Oriented Gradients) từ skimage
    hog_features, hog_image = hog(image_resized, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)

    # Kết hợp các đặc trưng từ Canny và HOG
    features = np.hstack((edges_flattened, hog_features))
    
    return features, edges, hog_image

# Hàm load dữ liệu từ thư mục dataset
def load_dataset(dataset_dir):
    X = []
    y = []
    classes = ['standing', 'lying', 'sitting']  # Các nhãn tư thế

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            features, _, _ = preprocess_image(image_path)
            X.append(features)
            y.append(class_idx)
    return np.array(X), np.array(y)

# Khởi tạo và huấn luyện mô hình SVM
def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', random_state=42)  # Dùng kernel tuyến tính
    svm_model.fit(X_train, y_train)
    return svm_model

# Khởi tạo và huấn luyện mô hình
dataset_dir = 'dataset'  # Đường dẫn đến thư mục chứa dữ liệu
X, y = load_dataset(dataset_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
svm_model = train_svm(X_train, y_train)

# Kiểm tra độ chính xác trên tập huấn luyện và kiểm tra
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Hàm dự đoán tư thế từ ảnh đầu vào và hiển thị kết quả
def predict_pose(model, image_path):
    features, edges, hog_image = preprocess_image(image_path)  # Trích xuất đặc trưng Edge + HOG
    prediction = model.predict([features])  # Dự đoán với SVM
    predicted_class = prediction[0]
    pose_labels = {0: "Standing", 1: "Lying", 2: "Sitting"}
    predicted_pose = pose_labels[predicted_class]

    # Đọc ảnh gốc và thay đổi kích thước ảnh cho vừa với màn hình
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (128, 128))  # Đảm bảo kích thước đồng đều

    # Hiển thị ảnh với Matplotlib
    plt.figure(figsize=(10, 6))

    # Hiển thị ảnh gốc
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Hiển thị ảnh phát hiện biên (Edge Detection)
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')

    # Hiển thị ảnh HOG
    plt.subplot(1, 3, 3)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.axis('off')

    # Thêm dự đoán và độ chính xác
    plt.suptitle(f"Predicted Pose: {predicted_pose} \n(Test Accuracy: {test_accuracy:.4f})", fontsize=16)
    plt.show()

# Chức năng chọn ảnh từ thư mục và dự đoán
def select_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        predict_pose(svm_model, image_path)

# Giao diện người dùng với Tkinter
root = tk.Tk()
root.title("Pose Recognition")
root.geometry("300x200")
btn_select_image = tk.Button(root, text="Select Image", command=select_image)
btn_select_image.pack(pady=20)
root.mainloop()
