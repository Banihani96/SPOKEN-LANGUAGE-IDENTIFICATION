# SPOKEN-LANGUAGE-IDENTIFICATION
The aim of this repository is to build 3 machine learning models that can identify the spoken language.
* The first model will convert Wav to a grey spectrogram and then apply CNN model
* The Second model will convert the Wav to an RGB Spectroram and then Apply CNN Model 
* The final model will extract 13 numerical features from a single wav and insert them in a  FFNN model 

The dataset used in the notebooks is based on Mozilla’s Common Voice. You will need to download the English V1 and the Arabic V7 datasets and then run the scripts.
The first scripts Data Pre-Processing will allow to split that data into training and testing and specify the amount of data taken from the original file.
