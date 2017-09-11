from __future__ import division, print_function, absolute_import
import tflearn
import os
import re
import sys
import wave

import numpy
import numpy as np
import skimage.io  # scikit-image
import librosa
import matplotlib
from random import shuffle
import tensorflow as tf

learning_rate = 0.0001
training_iters = 1  # steps
batch_size = 128

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits
path="C:\\Users\\aswin\\Big Data\\Final Project\\data\\spoken_numbers_pcm\\"
files=os.listdir(path)

def train_test_split(test_list):
  test_files=[]
  train_files=[]
  for wav in files:
      if any(key in wav for key in test_list):
        test_files.append(wav)
      else:
        train_files.append(wav)
  return test_files,train_files
  
test_files,train_files=train_test_split(['100','200','300','400'])
 

def dense_to_one_hot(labels_dense, num_classes=10):
  return numpy.eye(num_classes)[labels_dense]


def mfcc_train_batch_generator(train_files_list,batch_size=10):
  batch_features = []
  labels = []
  while True:
    print("loaded batch of %d files" % len(files))
    shuffle(train_files_list)
    for wav in train_files_list:
      if not wav.endswith(".wav"): continue
      wave, sr = librosa.load(path+wav, mono=True)
      label=dense_to_one_hot(int(wav[0]),10)
      labels.append(label)
      mfcc = librosa.feature.mfcc(wave, sr)
      mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
      batch_features.append(np.array(mfcc))
      if len(batch_features) >= batch_size:
        print("Loading a batch %d files" % len(labels))
        yield batch_features, labels  
        batch_features = []  
        labels = []


def mfcc_test_data(test_files_list):
  test_features = []
  test_labels = []
  shuffle(test_files_list)
  for wav in test_files_list:
      if not wav.endswith(".wav"): continue
      wave, sr = librosa.load(path+wav, mono=True)
      label=dense_to_one_hot(int(wav[0]),10)
      test_labels.append(label)
      mfcc = librosa.feature.mfcc(wave, sr)
      mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
      test_features.append(np.array(mfcc))
  return test_features, test_labels


test_X,test_Y=mfcc_test_data(test_files)
print(len(test_Y))
train_batch_gen=mfcc_train_batch_generator(train_files,64)

total_batches=np.ceil((len(files)-len(test_files))/64)
print(len(files),len(test_files),total_batches)




# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 

model = tflearn.DNN(net,tensorboard_dir="C:\\Users\\aswin\\Big Data\\Final Project\\tflogs", tensorboard_verbose=0)
for i in range(3000):
  train_X,train_Y=next(train_batch_gen)
  model.fit(train_X, train_Y, n_epoch=25, validation_set=(test_X, test_Y), show_metric=True,batch_size=batch_size)
#_y=model.predict(X)
model.save("C:\\Users\\aswin\\Big Data\\Final Project\\tflogs\\tflearn.lstm.model")
#print (_y)
#print (y)
