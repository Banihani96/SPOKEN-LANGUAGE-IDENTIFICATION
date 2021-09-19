from flask import Flask, render_template
import logging
from Prediction import ArabicEnglishClassifier
import numpy as np
from flask import flash


app = Flask(__name__)  
# define model path
model_path = './Model/CNN_GRAY_SCALE.h5'

# create instance
model = ArabicEnglishClassifier(model_path)
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=["GET", "POST"])
def index():

    return render_template("index.html", data="hey")

@app.route("/prediction", methods=["POST"])

def prediction():
    model.upload_file()
    model.audio_to_image_file("uploads/test.wav")
    image=model.load_image("uploads/test.wav.png")
    pred = model.model.predict(image)
    pred = np.argmax(pred)
    Results = ["Arabic","English"]
    return render_template("prediction.html", data=Results[pred])
   

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)



