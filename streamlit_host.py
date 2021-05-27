import streamlit as st
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os
import imutils
from tensorflow.python.keras.models import load_model

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        def detect_and_predict_mask(frame, faceNet, maskNet):
            # grab the dimensions of the frame and then construct a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

            faceNet.setInput(blob)
            detections = faceNet.forward()

            # initialize our list of faces, their corresponding locations and list of predictions

            faces = []
            locs = []
            preds = []

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    # we need the X,Y coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')

                    # ensure the bounding boxes fall within the dimensions of the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

                # only make a predictions if atleast one face was detected
                if len(faces) > 0:
                    faces = np.array(faces, dtype='float32')
                    preds = maskNet.predict(faces, batch_size=12)

                return (locs, preds)

        prototxtPath = os.path.sep.join([r'.\face_detector', 'deploy.prototxt'])
        weightsPath = os.path.sep.join(
            [r'.\face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        # Load trained deep learning model
        maskNet = load_model('face_mask_detection_system.h5')
        frame = imutils.resize(img, width=400)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corrosponding loactions

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we will use to draw the bounding box and text
            label = 'Mask' if mask > withoutMask else 'No Mask'
            color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

            # display the label and bounding boxes
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        return frame

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)






    