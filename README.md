# Time-Series Forecast Using LSTMs from Keras

## Overview

* In this project LSTM neural network has been used to forecast potential migraine attacks of patients suffering from it.
* The dataset that has been used is from the paper : https://www.ncbi.nlm.nih.gov/pubmed/26901341, Analysis of Trigger Factors in Episodic Migraineurs Using a Smartphone Headache Diary Applications.
* In this paper the authors made a smartphone app for migraine patients where they were required to enter all their migraine triggers along with any event of a migraine attack.
* 62 patients contributed for 4579 days of data for the research.

## Data Processing

* The dataset was first translated from korean language to English.
* After handling all the missing values and feature selection, the data had 1098 samples and 17 features.
* The selected features were sensitive to light, light/sound sensitive, stress, sleeping too much, lack of sleep, exercise, physical fatigue, weather/temperature changes, excessive sunlight, noise,   excessive drinking, irregular eating, excessive caffeine, excessive smoking, chocolate cheesecake, travel and migraine presence.
* The target variable which should be migraine presence was shifted by one step backwards so as to forecast one step forward.
* 1000 samples were used for training and 50 for validation which were selected randomly from the dataset.

## Model

* Long Short Term Memory(LSTM) Neural Networks were used for the purpose of keeping track of patients' behaviors and patterns which would influence the forecasts.
* The LSTM has 50 units with a dense layer. The cost function is mean absolute error and optimization algorithm is Adam.
* A batch size of 50 was used with 100 iteraions.

## Result

* The mean squared error between the actual values and predicted values was 27.59 percent which is quite decent given the small size of the training data.



 
