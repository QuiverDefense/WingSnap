from flask import Flask, render_template, request

from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.models import Model
import numpy as np

app = Flask(__name__)
model = load_model("LepModel.h5")

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    result = np.argmax(model.predict(image), axis = -1)
    if str(result) == "[0]":
        result = "Atlas Moth Adult"
    if str(result) == "[1]":
        result = "Atlas Moth Larva"
    if str(result) == "[2]":
        result = "European Skipper Adult"
    if str(result) == "[3]":
        result = "European Skipper Larva"    
    if str(result) == "[4]":
        result = "Isabella Tiger Moth Adult"    
    if str(result) == "[5]":
        result = "Isabella Tiger Moth Larva"    
    if str(result) == "[6]":
        result = "Lime Butterfly Adult"    
    if str(result) == "[7]":
        result = "Lime Butterfly Larva"    
    if str(result) == "[8]":
        result = "Monarch Butterfly Adult"    
    if str(result) == "[9]":
        result = "Monarch Butterfly Larva"    


    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
# run with flask run -h 192.168.0.16 where 192.168.0.16 is your machine's IPv4 address