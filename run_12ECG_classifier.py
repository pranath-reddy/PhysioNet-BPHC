#!/usr/bin/env python

import numpy as np
from get_12ECG_features import get_12ECG_features
from tensorflow.keras.models import load_model
from scipy.stats import mode

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(9, dtype=int)
    current_score = np.zeros(9)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(-1,1,4500)
    print(features.shape)
    #label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)
    score = np.mean(score, axis=0)
    
    pred_labels = model.predict_classes(feats_reshape)
    pred_label = mode(pred_labels)[0][0]
    print(pred_label)
    
    for i in range(9):
        current_score[i] = np.array(score[i])
        
    current_label[pred_label] = 1
        
    for i in range(num_classes):
        if score[i] > 0.5:
            current_label[i] = 1
    
    return current_label, current_score

def load_12ECG_model():

    # load the model from disk
    loaded_model = load_model('classifier.h5')

    return loaded_model
