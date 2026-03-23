import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("📷 简易人脸识别应用")

# 加载 OpenCV 自带的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 上传图片
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 转成 PIL 图像
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 转成灰度图（方便检测）
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # 在图上画框
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # 显示结果
    st.subheader(f"检测到 {len(faces)} 张人脸")
    st.image(img_array, caption="人脸识别结果", use_column_width=True)