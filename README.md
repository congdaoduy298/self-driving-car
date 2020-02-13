# self-driving-car
Use CNN to predict steering wheel angles of Udacity Self-Driving Car Engineer - https://github.com/udacity/self-driving-car-sim
In training mode of simulator: we manually drive vehicel and collect data.

The project consists of 4 files:
+ data_augment.py : image horizontal flip, random translation, random shadow, random brightness, ... . This file prevent overfitting.
+ train_model_track1.ipynb : using data from trainning mode of track 1 to train model_track1 with CNN. Normalize input data -> 3 Convolution 5X5 kernel, stride 2X2 -> 2 Convolution 3X3 kernel -> Fully Connected -> Output: Vehicel control.
+ train_model_track2.ipynb: using model from track1 as pretrained-model and data from training mode of track 2 to train model_track2 with the same CNN before.
+ drive.py : get image in game -> preprocessing image -> predict steering angle -> send value of steering angle and speed to drive car.

Reference:

https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
https://nttuan8.com/bai-8-o-to-tu-lai-voi-udacity-open-source/
