#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from joblib import dump, load
from tensorflow.keras.models import load_model

def run_12ECG_classifier(data,header_data,classes,model):

    #num_classes = len(classes)
    #current_label = np.zeros(num_classes, dtype=int)
    #current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class.
    features=np.asarray(get_12ECG_features(data,header_data,model[1]))
    preds = model[0].predict(features).toarray()
    probas = model[0].predict_proba(features).toarray()
    
    pred_vals = np.mean(preds, axis=0)
    current_score = np.mean(probas, axis=0)
    current_label = np.zeros(9, dtype=int)
    
    for i in range(pred_vals.shape[0]):
        if pred_vals[i] > 0.2:
            current_label[i] = 1

    return current_label, current_score

def load_12ECG_model():

    # load the model from disk
    loaded_model = load('classifier.joblib')
    fe_model = load_model('./FE_Model.h5')

    return [loaded_model, fe_model]
