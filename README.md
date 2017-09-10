# Spoken-Number-Recognition

## Summary:
The goal of this project is to identify individually spoken numbers (0-9) using Long Short Term Memory (LSTM) network, a variant of recurrent neural network(RNN) using Tensorflow API.  The project explores the use of uni-directional and bi-directional LSTM in solving the problem.

## The Dataset:
The source dataset for this problem is downloaded from http://pannous.net/files/spoken_numbers.tar. This dataset is a collection of 2402 .wav files of spoken numbers ranging from 0-9 by 15 different speakers comprising of both male and female voices. The dataset is diverse as it covers 16 different tone at which a number is spoken by each of the speakers. 

## RNN:
They are the most widely used sequence to sequence neural network architectures with applications in many natural language processing problems such as language translation, speech recognition etc. 

## Why RNN ?
•	RNN solves the fundamental drawbacks of many neural network architectures – persistent memory. RNN just like humans have the capacity to remember or persist the information for longer duration.
•	This capability of RNN makes it a natural choice to model human languages where the occurrence of words is highly dependent on those that came before.

## Drawbacks:
•	The network can become complex especially the bi-directional RNN’s. Hence the understanding of the back propagation and the temporal nature of RNN could be difficult.
•	The model takes several hours to train especially if the intention to make it generic for any speaker. 

## Demo & Results:
•	The project demonstrates the following:
•	Feature extraction from raw audio(.wav) files including padding and transformations.
•	Single feed-forward LSTM network for speech recognition.
•	Bi-directional (forward and backward) LSTM network for speech recognition.
•	Tensorboard graphs, accuracy, loss and weights histogram on training, validation datasets. 
