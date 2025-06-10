import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset_path = r"C:\Users\lsj24\OneDrive\Desktop\pc_diagnose/image"
folders = ["autumnwarm", "springwarm", "summercool", "wintercool"]

X = []
y = []

def get_avg_b_value(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_b = np.mean(lab[:, :, 2])
    return avg_b

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.exists(folder_path):
        print(f"{folder_path} 경로가 존재하지 않습니다.")
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(folder_path, file)
            avg_b = get_avg_b_value(file_path)
            if avg_b is not None:
                X.append([avg_b])
                y.append(folder)

# 레이블 인코딩 (문자열 폴더명 → 숫자 라벨)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("샘플 개수:", len(X))
print("레이블 클래스:", le.classes_)

# X, y_encoded가 머신러닝 모델 학습용 데이터셋이 됩니다.
