from flask import Flask, request, render_template, Response
import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import json

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
LEFT_CHEEK = [234, 93, 132, 58, 172, 136]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365]

model = joblib.load('model.pkl')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark

        def get_points(index_list):
            return np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in index_list])

        left_cheek_pts = get_points(LEFT_CHEEK)
        right_cheek_pts = get_points(RIGHT_CHEEK)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [left_cheek_pts], 255)
        cv2.fillPoly(mask, [right_cheek_pts], 255)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        cheek_pixels = lab[mask == 255]

        if cheek_pixels.size == 0:
            return None

        mean_vals = np.mean(cheek_pixels, axis=0)
        std_vals = np.std(cheek_pixels, axis=0)

        features = np.concatenate([mean_vals, std_vals])
        return features.reshape(1, -1)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return Response(json.dumps({"error": "파일이 업로드되지 않았습니다."}, ensure_ascii=False), mimetype='application/json'), 400
    file = request.files['file']
    if file.filename == '':
        return Response(json.dumps({"error": "파일명이 없습니다."}, ensure_ascii=False), mimetype='application/json'), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    features = preprocess_image(file_path)
    os.remove(file_path)

    if features is None:
        return Response(json.dumps({"error": "얼굴 인식에 실패했습니다. 올바른 얼굴 사진을 올려주세요."}, ensure_ascii=False), mimetype='application/json'), 400

    prediction = model.predict(features)[0]
    response_data = {"predicted_personal_color": prediction}
    response_json = json.dumps(response_data, ensure_ascii=False)

    return Response(response_json, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
