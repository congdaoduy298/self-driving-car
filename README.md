# self-driving-car
Use CNN to predict steering wheel angles of Udacity Self-Driving Car Engineer - https://github.com/udacity/self-driving-car-sim
In training mode of simulator: we manually drive vehicel and collect data.

The project consists of 4 files:
+ data_augment.py : image horizontal flip, random translation, random shadow, random brightness, ... . This file prevent overfitting.
+ train_model_track1.ipynb : using data from tranning mode to train model
