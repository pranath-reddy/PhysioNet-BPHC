# Python classifier for the PhysioNet/CinC Challenge 2020

## Contents

This classifier uses three scripts:

* `run_12ECG_classifier.py` makes the classification of the clinical 12-Leads ECG. Add your classification code to the `run_12ECG_classifier` function. It calls `get_12ECG_features.py` and to reduce your code's run time, add any code to the `load_12ECG_model` function that you only need to run once, such as loading weights for your model.
* `get_12ECG_features.py` extract the features from the clinical time-series data. This script and function are optional, but we have included it as an example.
* `driver.py` calls `load_12ECG_model` once and `run_12ECG_classifier` many times. Both functions are in `run_12ECG_classifier.py` file. This script also performs all file input and output. Please **do not** edit this script or we may be unable to evaluate your submission.

## Usage

You can run this classifier by installing the packages in the `requirements.txt` file and running

    python driver.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output classification files. The PhysioNet/CinC 2020 webpage provides a training database with data files and a description of the contents and structure of these files.
