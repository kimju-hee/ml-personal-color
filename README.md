# 🌈 머신러닝 기반 퍼스널컬러 분류 웹사이트 만들기


<br>

## 프로젝트 소개
사진을 올리면 내 퍼스널컬러를 딱! 알려주는 웹 애플리케이션이에요.  
단순 색상 평균이 아니라, 머신러닝 모델이 똑똑하게 분석해준답니다!

---


<br>

## ✨ 주요 기능
- 💻 이미지 업로드 및 저장  
- 🎨 OpenCV로 이미지 전처리 (LAB 색공간 B 채널 평균 추출)  
- 🌲 Random Forest 분류기로 퍼스널컬러 예측  
- 🏆 상위 2개 결과와 확률 보여주기  
- 🌐 Flask 기반 웹 서비스 제공


---


<br>

## 🛠️ 사용 기술 스택
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-FF6C37?style=for-the-badge&logo=python&logoColor=white)



---

<br>

## 📊 데이터와 모델
- **데이터셋**: `autumnwarm`, `springwarm`, `summercool`, `wintercool` 각 폴더별 이미지  
- **특징 추출**: LAB 색 공간에서 B 채널 평균값만 사용  
- **모델**: RandomForestClassifier  
- **성능**: 약 76% 이상의 정확도 달성!



---


<br>

## 🔍 결과와 개선점
- 예상보다 높은 정확도를 나타냈습니다.
- 하지만 가을 톤(autumnwarm) 같은 일부 클래스 정확도가 낮아서 데이터 보강과 전처리 개선이 필요합니다.

![output](https://github.com/user-attachments/assets/62919b51-bc0c-4ecc-b868-780146f2a901)


---


<br>

## 🚀 사용 방법
1. 저장소 클론 후 라이브러리 설치  
2. 데이터셋 준비 또는 학습된 모델 다운로드  
3. `python app.py`로 Flask 서버 실행  
4. 웹 브라우저에서 [http://localhost:5000](http://localhost:5000) 접속  
5. 이미지 업로드하고 퍼스널컬러 진단 끝!  

