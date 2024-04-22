import streamlit as st
import cv2
from paddleocr import PaddleOCR  # Ensure PaddleOCR is installed
import google.generativeai as genai  # Make sure this is correctly installed and configured
import numpy as np
import tempfile
import os
GOOGLE_API_KEY='YOUR API KEY'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize OCR and text generation model
ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = genai.GenerativeModel(model_name = "gemini-pro")


def ocr_response(image):
    # Convert the image to RGB (PaddleOCR requirement)
    result = ocr.ocr(image, cls=True)
    strings = [entry[1][0] for sublist in result for entry in sublist]
    concatenated_string = ", ".join(strings)
    prompt = f"Write the drugs or medicine names that are present in this: {concatenated_string}"
    answer = model.generate_content(prompt)
    return answer.text

# Streamlit UI
st.title('OCR-based Drug Name Extractor')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    opencv_image = cv2.imread(tmp_file_path)

    st.image(opencv_image, channels="BGR", caption="Uploaded Image")

    if st.button('Extract and Generate'):
        with st.spinner('Processing...'):
            response = ocr_response(opencv_image)
            st.markdown(response)
else:
    st.write("Upload an image to get started.")

