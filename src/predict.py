# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 11.12.19 13:40

import joblib
import sounddevice as sd
from scipy.io.wavfile import write
from src.emrec_model import extract_feature
import numpy as np


def record_voice():
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('recorded_data/output.wav', fs, myrecording)  # save as WAV file
    return myrecording


def predict(model_path, test_file):
    test = extract_feature(test_file, mfcc=True, chroma=True, mel=True)
    test = np.array(test)
    model = joblib.load(model_path)
    res = model.predict(test)
    return res


if __name__ == '__main__':

    print("Recording is started.......")
    record = record_voice()
    print("File is stored at recorded folder.")
    print("Prediction of is processed!")
    res = predict(model_path="models/best_model.sav", test_file="recorded_data/output.wav")
    print("Predicted emotion :", res)
