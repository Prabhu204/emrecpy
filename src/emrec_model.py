# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 11.12.19 21:03

import librosa
import soundfile
import os,glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import argparse


# DataFlair - Extract features (mfcc, chroma, mel) from a sound file


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        # print(sample_rate)
        if chroma:
            stft = np.abs(librosa.stft(X))
            # print(librosa.stft(X).shape)
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        # print(result.shape)
    return result



#DataFlair - Emotions in the RAVDESS dataset


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']


# DataFlair - Load the data and extract features for each sound file

def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("data/ravdess-data/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue # continue will jump back to the top of the loop. pass will continue processing.
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
        # train_test_split(np.array(x), y, test_size=test_size, random_state=9)
        # print(np.array(x).shape)
    # return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
    print("Data has been loaded.......")
    return np.array(x), y


def train(test_size=True, save_model= False, plot_fig= True):

    global x_train, y_train, y_test, x_test

    X,y = load_data(test_size=0.2)
    if test_size:
        x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    layer_size= [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    scores = []
    models = []
    for size in layer_size:
        clf= MLPClassifier(alpha=0.001, batch_size=256, epsilon=1e-08,
                             hidden_layer_sizes=(size,), learning_rate='adaptive', max_iter=1000, verbose=True)
        if test_size:
            model = clf.fit(x_train,y_train)
            models.append(model)
            ac = accuracy_score(y_test, model.predict(x_test))
            scores.append(ac)
        else:
            score = cross_val_score(clf, X, y, cv=10)
            scores.append(score.mean())
            model = clf.fit(X, y)
            models.append(model)
    if plot_fig:
        sns.lineplot(layer_size, scores)
        plt.title("Model Performance")
        plt.xlabel("Hidden layer size")
        plt.ylabel("Accuracies")
        plt.savefig("plots/model_performance.png")
        plt.close()

    print("All accuracies:", scores)
    best_hlayer_size = layer_size[scores.index(max(scores))]
    print("Best hidden layer neuron size:", best_hlayer_size)
    max_acc = max(scores)
    print("Best accuracy:", max_acc)
    if save_model:
        print("Saving best model.......")
        best_model = models[scores.index(max(scores))]
        joblib.dump(best_model, 'models/best_model.sav')
    with open("output.txt", "w") as f:
        f.write("All accuracies: {} \nBest hidden layer neuron size: {} \nBest accuracy achieved :{}"
                .format(scores, best_hlayer_size, max_acc))

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion detection from audio file')
    parser.add_argument("--test_size", action='store_true')
    parser.add_argument("--plot",  action='store_true')
    parser.add_argument("--save_model", action='store_false')
    args = parser.parse_args()
    sc = train(test_size=args.test_size, save_model=False)
    print(sc)