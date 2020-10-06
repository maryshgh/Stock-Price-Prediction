# Stock-Price-Prediction
In this project, the recurrent neural networks are used for stock price prediction.

This is a code package related to the following project. In this project, the goal is to predict the Apple stock price using the time series prediction approach. Th variable "Recurrence_period" defines the timesteps of recurrence and "Prediction_period" defines the number of days we are predicting. A deep LSTM network is used for the time series prediction. The network contains 3 hidden layers, first hidden layer with 100 neurons, second and third hidden layers with 50 neurons and output layer with just one neuron. Furthermore, for training, the 'adam' optimizer has been used which minimizes the loss which is mean squared error for this regression problem.

This repository contains the Python code required to reproduces all the numerical results.

## Content of Code Package
The package contains one Python file including Python code and one comma seperated file including 5 years of stock prices.
