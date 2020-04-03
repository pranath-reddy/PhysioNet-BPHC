#!/usr/bin/env python

import numpy as np
from get_12ECG_features import get_12ECG_features
from tensorflow.keras.models import load_model

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    #feats_reshape = features.reshape(1,-1)
    #label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])
        
    for i in range(num_classes):
        if score[0][i] > 0.5:
            current_label[i] = 1

    return current_label, current_score

def load_12ECG_model():

    # load the model from disk
    loaded_model = load_model('classifier.h5')

    return loaded_model
