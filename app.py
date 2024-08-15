import streamlit as st

import cv2
from PIL import Image

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import time

# load model
@st.cache_resource
def loadModel():
    model = load_model("CNNModel_fer_3emo.h5")
    return model

model = loadModel()

# load face cascade
try:
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    st.write('Error loading face cascade classifier')


def preprocess(grayImg, picSize=48):

    pred = 'snake'

    # cvgrayImg is treated like an numpy array (https://rb.gy/79e2f)
    faces = faceCascade.detectMultiScale(grayImg)

    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(grayImg, (x, y), (x+w, y+h),
                          color=(255, 0, 0), thickness=1)

         # Extract region of interest & resize it
            roiGray = grayImg[y: y + h, x: x + w]
            roiGray = cv2.resize(roiGray, (picSize, picSize),
                                 interpolation=cv2.INTER_AREA)

            # Preprocess the ROI (normalize -> convert to array)
            flagPre = False
            if np.sum([roiGray]) != 0:
                roiGray = roiGray.astype('float')/255.0
                roiGray = img_to_array(roiGray)
                roiGray = np.expand_dims(roiGray, axis=0)
                flagPre = True
# ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            if flagPre:
                # Perform prediction on the batch (size 1)
                pred = model.predict(roiGray)[0]
                print(pred)
                pred = int(np.argmax(pred))
                pred = ['anger', 'disgust', 'fear', 'happy',
                        'neutral', 'sad', 'surprise'][pred]

            labelPos = (x, y)
            cv2.putText(grayImg, pred, labelPos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    else:
        pred = "No face recognized"
        return grayImg, pred
    return grayImg, pred

# 4.6689624e-01 1.1547882e-04 3.9052553e-02 2.3853797e-03 6.0666800e-02 4.2035532e-01 1.0528210e-02]

# Capture snapshot using st.camera_input
checkbox_value = st.checkbox("Capture Snapshot!", key = "snapshot_checkbox")

pred = "No face recognized"

if checkbox_value:
    
    picture = st.camera_input("Capture Snapshot")
    consent = True

    if picture is not None and consent == True:

        # opening image in PIL format
        grayImg = Image.open(picture)

        # converting image to CV format in black & white (https://rb.gy/ha8bl)
        cvgrayImg = cv2.cvtColor(np.array(grayImg), cv2.COLOR_RGB2GRAY)

        # preprocess the image & make predictions
        cvgrayImg, pred = preprocess(cvgrayImg)

        st.write(f'{pred}')
        st.image(cvgrayImg)
        
        time.sleep(5)
        
        # Display the processed image
        st.write(f"Emotion detected: {pred}")

        # Take input from user
        lang = st.text_input('Language')
        singer = st.text_input("Singer")

        if lang and singer:
            st.write(
                f"Searching for {lang} songs by {singer} related to {pred} emotion...")

            # Construct the YouTube search query
            query = f"{lang} {singer} {pred} song"

            # Search for the query on YouTube
            youtube_search_url = "https://www.youtube.com/results?search_query="
            youtube_query_url = youtube_search_url + query.replace(" ", "+")
            st.write(f"Songs for your mood: [link]({youtube_query_url})")

        consent = False


