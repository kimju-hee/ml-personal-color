import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'app/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load('./notebooks/sklearn_personal_color_model.pkl')
le = joblib.load('./notebooks/sklearn_label_encoder.pkl')

def extract_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (100, 100))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_avg = np.mean(img_lab[:, :, 2])
    return [b_avg]

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(img_path)

            feature = extract_feature(img_path)
            if feature:
                probs = model.predict_proba([feature])[0]  # 각 클래스 확률
                pred = np.argmax(probs)
                label = le.inverse_transform([pred])[0]

                # 2등 확률과 라벨 찾기
                sorted_indices = np.argsort(probs)[::-1]
                second_idx = sorted_indices[1]
                second_label = le.inverse_transform([second_idx])[0]
                second_prob = probs[second_idx]

                image_url = '/static/uploads/' + filename
                return render_template(
                    'result.html',
                    label=label,
                    image_path=image_url,
                    prob=probs[pred],
                    second_label=second_label,
                    second_prob=second_prob
                )
            else:
                return "이미지 처리에 실패했습니다."
        else:
            return "파일이 업로드되지 않았습니다."
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
