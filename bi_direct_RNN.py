import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
import os
import re
import sys
import wave
import pandas as pd
import numpy
import skimage.io  # scikit-image
import librosa
import matplotlib
from random import shuffle



# Parameters
learning_rate = 0.001
training_iters = 750
batch_size = 128
display_step = 10

# Network Parameters
n_input = 80 
n_steps = 20 
n_hidden = 350 # LSTM layer num of features
n_hidden2 = 1024 # hidden layer num of features
n_classes = 10 

#hparam="BiRNN_FC"+str(n_hidden)+"_"+str(learning_rate)+"_"+str(batch_size)
hparam="Output2"




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
    #print("loaded batch of %d files" % len(files))
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
        #print("Loading a batch %d files" % len(labels))
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
train_batch_gen=mfcc_train_batch_generator(train_files,batch_size)



 

# tf Graph input
with tf.name_scope("Inputs"):
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("float", [None, n_classes])


# Define weights

with tf.name_scope("RNN"):
	rnn_weight= tf.Variable(tf.random_normal([2*n_hidden, n_hidden2]),name="W")
	rnn_bias= tf.Variable(tf.random_normal([n_hidden2]),name="B")

with tf.name_scope("FC"):
	fc_weight= tf.Variable(tf.random_normal([n_hidden2, n_classes]),name="W")
	fc_bias= tf.Variable(tf.random_normal([n_classes]),name="B")


def BiRNN(x):

	# Prepare data shape to match `bidirectional_rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, n_steps, 1)

	# Define lstm cells with tensorflow
	# Forward direction cell
	lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	# Backward direction cell
	lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
						  dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	l1= tf.matmul(outputs[-1], rnn_weight) + rnn_bias
	rnn1=tf.nn.dropout(l1,0.8)
	fc1=tf.matmul(rnn1,fc_weight)+fc_bias
	
	tf.summary.histogram("weights", rnn_weight)
	tf.summary.histogram("biases", rnn_bias)
	tf.summary.histogram("FC weights", fc_weight)
	tf.summary.histogram("FC biases", fc_weight)
	return fc1

pred = BiRNN(x)


# Define loss and optimizer
with tf.name_scope("Loss"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name="Loss")
	tf.summary.scalar("Loss",cost)

with tf.name_scope("Test_Loss"):
	test_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name="Test_Loss")
	ltst=tf.summary.scalar("Test_Loss",test_cost)	
	
with tf.name_scope("Train"):	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("Accuracy"):
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="Train_Accuracy")
	tf.summary.scalar("Accuracy",accuracy)

with tf.name_scope("Test_Accuracy"):
	correct_test_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	test_accuracy = tf.reduce_mean(tf.cast(correct_test_pred, tf.float32),name="Test_Accuracy")
	atst=tf.summary.scalar("Test_Accuracy",test_accuracy)
	
	
# Initializing the variables
init = tf.global_variables_initializer()

summ = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("C:\\Users\\aswin\\Big Data\\Final Project\\tflogs2\\"+hparam)
    writer.add_graph(sess.graph)
    
    step = 1
    # Keep training until reach max iterations
    while step <=training_iters:
        batch_x, batch_y = next(train_batch_gen)
        # Run optimization op (backprop)
        loss,_,train_acc,s=sess.run([cost,optimizer,accuracy,summ], feed_dict={x: batch_x, y: batch_y})
        writer.add_summary(s,step)
        test_pred,test_loss,test_acc,acc_summ,loss_summ = sess.run([pred,test_cost,test_accuracy,atst,ltst],feed_dict={x: test_X, y: test_Y})
        writer.add_summary(acc_summ,step)
        writer.add_summary(loss_summ,step)
        if step %100 == 0:
            numpy.savetxt("pred.csv", test_pred, delimiter=",")
            numpy.savetxt("actual.csv", test_Y, delimiter=",")
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Batch Accuracy= " + \
                  "{:.5f}".format(train_acc))
            print("Iter " + str(step) + ", Test Loss= " + \
	          "{:.6f}".format(test_loss) + ", Test Accuracy= " + \
                  "{:.5f}".format(test_acc))
        step += 1
    print("Optimization Finished!")

 