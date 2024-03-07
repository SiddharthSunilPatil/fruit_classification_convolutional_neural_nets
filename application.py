from flask import Flask, request, render_template, jsonify
import os
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from src.logger import logging
import keras
import base64
from src.pipelines.predict_pipeline import PredictPipeline



application=Flask(__name__)

app=application

# defining class names
class_names=['FreshApple','FreshBanana','FreshGrape','FreshGuava',
             'FreshJujube','FreshOrange','FreshPomegranate','FreshStrawberry',
             'RottenApple','RottenBanana','RottenGrape','RottenGuava',
             'RottenJujube','RottenOrange','RottenPomegranate','RottenStrawberry']

@app.route('/')
def index():
    return render_template('index.html')

#defining function for GET and POST methods
@app.route('/classify_image',methods=['GET','POST'])
def ClassifyImage():
    if request.method=="GET":
        return render_template('home.html')
    else:


        logging.info("Entered the post method")
        #checking if request contains image file
        if 'file' not in request.files:
            return jsonify({'error':'No file part in the request'})
        
        file = request.files['file']
        logging.info("File requested successfully")

        # confirming file is an image
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        
        #reading the image file
        image_data=file.read()
        logging.info("File read successfully")

        #saving file to temp folder
        #if file:
           #filename = file.filename
            #file.save(os.path.join('temp', filename))

        # readying the image for prediction
        image=Image.open(io.BytesIO(image_data))

        #temporarily saving the image
        img_bytes=io.BytesIO()
        image.save(img_bytes,format='PNG')
        img_bytes=img_bytes.getvalue()

        #Encoding image bytes to base64
        img_base64=base64.b64encode(img_bytes).decode('utf-8')

        #Expanding dimensions
        image=np.expand_dims(image,axis=0)
        logging.info("Image transformation completed")

        # predicting / classifying the image
        predictor=PredictPipeline()
        predictions=predictor.predictdata(image)

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.round(100*(np.max(predictions[0])),2)
        logging.info("Prediction successful")


        
        return (
            render_template("home.html",class_name=predicted_class,confidence=confidence,read_img=img_base64)
            #render_template('home.html',class_name=predicted_class,confidence=float(confidence))
        )

if __name__=="__main__":
    app.run(host='0.0.0.0')
