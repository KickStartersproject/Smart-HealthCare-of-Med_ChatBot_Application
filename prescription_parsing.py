
import cv2
import pathlib
import textwrap
import re
import os
import string
import nltk
import gradio as gr
import google.generativeai as genai
from IPython.display import display, Markdown
from paddleocr import PaddleOCR,draw_ocr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from google.colab.patches import cv2_imshow

nltk.download('stopwords')
nltk.download('punkt')

def clean_string(text):
    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(tokens)

    return cleaned_text


def pipeline(img_path):
  result = ocr.ocr(img_path, cls=True)
  strings = [entry[1][0] for sublist in result for entry in sublist]


  concatenated_string = ""
  for string in strings:
      #print(string)
      concatenated_string += string + " , "
  response = model.generate_content(f"Name some drugs present in this string: ${concatenated_string}?")
  return clean_string(response.text)




ocr = PaddleOCR(use_angle_cls=True, lang='en') 

img_path = 'prescription.png'
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

result1 = ocr.ocr(img_path, cls=True)
cnt = 0
for line in result1:
    print(line)
    cnt += 1


strings = [entry[1][0] for sublist in result1 for entry in sublist]


concatenated_string = ""
for string in strings:
    print(string)
    concatenated_string += string + " , "

YOUR_API_KEY = ''
key = YOUR_API_KEY
genai.configure(api_key = key)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content(f"Name some drugs present in this string: ${concatenated_string}?")


answer = pipeline('/content/prescription.png')

print(answer)

