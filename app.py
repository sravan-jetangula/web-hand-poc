import streamlit as st
import cv2
import numpy as np

st.title("Hand Image Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(img, channels="BGR", caption="Original Image")
    st.image(gray, channels="GRAY", caption="Grayscale Image")
