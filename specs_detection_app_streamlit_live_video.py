import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageEnhance
import os
import mtcnn
import matplotlib.pyplot as plt
import time

face_detector = mtcnn.MTCNN()
specs_model=load_model("specs_detector.h5")

#st.title("Spectacles Detection in Live Video Stream")
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Spectacles Detection in Live Video</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

st.text("")

#run = st.button('Start')
# if run:
#     st.write("Live Video Stream starts...................")


if 'running' not in st.session_state:
    st.session_state.running = False

def start_stream():
    st.session_state.running = True

def stop_stream():
    st.session_state.running = False

start_button = st.button("Start Video Stream")
stop_button = st.button("Stop Video Stream")

if start_button:
    start_stream()

if stop_button:
    stop_stream()
    st.write("Video Stream stopped...................")


try:
    
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)      
    
    while st.session_state.running == True:
        
        ret, img = camera.read()
            
        org1 = (25, 25)
        org2 = (55, 55)
    
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        cv2.imwrite("test.jpg", img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = plt.imread("test.jpg")
        faces = face_detector.detect_faces(pixels)
        (x, y, w, h) = tuple(faces[0]['box'])
        face_img = img_rgb[y:y + h, x:x + w] 
        
        test_image = cv2.resize(face_img,(128,128))
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image/255    
        result = specs_model.predict(test_image)
        
        if result[0][0] > 0.05:
            prediction = 'No Spectacles'
            confidence_score = result[0][0]
            confidence_score = round(confidence_score, 4)
        else:
            prediction = 'Spectacles Found'
            confidence_score = (1.0 - result[0][0])
            confidence_score = round(confidence_score, 4)
    
        cv2.putText(img_rgb, str(prediction), org1, font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img_rgb, "Confidence: "+str(confidence_score), org2, font, 1, (0, 0, 255), 3, cv2.LINE_AA)    
        
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break

        FRAME_WINDOW.image(img_rgb)
        


    camera.release()
    cv2.destroyAllWindows()
    
        # cv2.imshow('Specs Detection', img)
        # if cv2.waitKey(1) == ord('q'):
        #     break
    
    # else:
    #     st.write('Stopped')

except:
    st.write("Live Video Stream stops...................")


