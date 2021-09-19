# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 12:11:51 2021

@author: Ahmad.Ibrahim
"""

import numpy as np
import imageio
import warnings
import librosa as lr
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import logging


UPLOAD_FOLDER = '/uploads'
sample_rate = 8000
image_width = 150
image_height = 150
ALLOWED_EXTENSIONS = {'wav'}

class ArabicEnglishClassifier:
    
        
    def allowed_file(self,filename):
        return '.' in filename and \
                   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def __init__(self, model_path):
            logging.info("Language Detection from Speech Task")
            self.model = load_model(model_path)
            logging.info("Model is loaded!")
    

    def upload_file(self):
     if request.method == 'POST':
         # check if the post request has the file part
         if 'file' not in request.files:
             flash('No file part')
             return redirect(request.url)
         file = request.files['file']
         # if user does not select file, browser also
         # submit an empty part without filename
         if file.filename == '':
             flash('No selected file')
             return redirect(request.url)
         if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save("uploads/test.wav")
             #return redirect(url_for('uploaded_file',filename=filename))
    
    def load_audio_file(self,audio_file):
        warnings.simplefilter('ignore', UserWarning)
        
        audio_segment, _ = lr.load(audio_file, sr=8000)
        return audio_segment
    
        warnings.simplefilter('default', UserWarning)
        
    def fix_audio_segment_to_10_seconds(self,audio_segment):
        target_len = 10 * 8000
        audio_segment = np.concatenate([audio_segment]*3, axis=0)
        audio_segment = audio_segment[0:target_len]
        
        return audio_segment
     
    
    def spectrogram(self,audio_segment):
        # Compute mel-scaled spectrogram image
        hl = audio_segment.shape[0] // image_width
        spec = lr.feature.melspectrogram(audio_segment, n_mels=image_height, hop_length=int(hl))
    
        # Logarithmic amplitudes
        image = lr.core.power_to_db(spec)
    
        # Convert to numpy matrix
        image_np = np.asmatrix(image)
    
        # Normalize and scale
        image_np_scaled_temp = (image_np - np.min(image_np))
        
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
    
        return image_np_scaled[:, 0:image_width]
    
    def load_image(self,filename):
        # load the image
        img = load_img(filename, grayscale=True, target_size=(150, 150))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 1 channel
        img = img.reshape( 1,150, 150,1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img
    
    
    def audio_to_image_file(self,audio_file):
        out_image_file = audio_file + '.png'
        audio = self.load_audio_file(audio_file)
        audio_fixed = self.fix_audio_segment_to_10_seconds(audio)
        if np.count_nonzero(audio_fixed) != 0:
            spectro = self.spectrogram(audio_fixed)
            imageio.imwrite(out_image_file, spectro)
        else:
            print('WARNING! Detected an empty audio signal. Skipping...')    
            
    def main():
    	model = ArabicEnglishClassifier('CNN GRAY SCALE.h5')
    	predicted_class = model.prediction()
    	logging.info("This is an image of a {}".format(predicted_class)) 
    
    
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        main()
