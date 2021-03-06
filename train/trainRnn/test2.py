import json
import numpy as np
import random
import collections
import time
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import cv2
from random import shuffle
start_time = time.time()

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

n_input = 15
training_iters = 500000
n_hidden = 15
learning_rate = 0.001

def rnn(x, weight, bias):
    '''
     define rnn cell and prediction
    '''

    x = tf.reshape(x, [-1, n_input * 128])
    x = tf.split(x, n_input, 1)
    

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

acc_total = 0
loss_total = 0
display_step = 1000

pathFloderStore = '/home/balienig/Documents/Git/PJ-cPeople/train/trainRnn/imageTrain/walk/'
Im_Height = 75
Im_Width = 30
listName = []
ImageSize = 50
sequence_length = 15
for folder , dirs, files in os.walk(pathFloderStore):
    nameFolder = folder.split('/')
    if nameFolder[10] != "" :
        listName.append(nameFolder[10])

############################################
print(listName)
listLabel = []
numClass = len(listName)
for i in range(numClass):
    listLabel.append([0]*numClass)
    listLabel[i][i] = 1
# print(listLabel)
training_data = []
numRnn = int(((((Im_Height-1)/2)-1)/2)*((((Im_Width-1)/2)-1)/2))
for folder , dirs, files in os.walk(pathFloderStore):
	for file in files:
		path = os.path.join(folder,file)
		word_label = path.split('/')
		index = listName.index(word_label[10])
		# print(index)
		# print(word_label[10])
		img = cv2.imread(path)
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img,(Im_Height,Im_Width))
		# print(img.shape)
		training_data.append([np.array(img),np.array(listLabel[index])])
        # shuffle(training_data)

###############################################
    
x = tf.placeholder("float", [None, n_input, 128])
y = tf.placeholder("float", [None, numClass])

weight = tf.Variable(tf.random_normal([n_hidden, numClass]))
bias = tf.Variable(tf.random_normal([numClass]))

logits = rnn(x, weight, bias)
prediction = tf.nn.softmax(logits)
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(softmax)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

allresult = []
with tf.Session() as session:
    
    session.run(init_op)

    #writer.add_graph(session.graph)
    
    for i in range(0,len(train_data)):
        
        input_data = np.reshape(train_data[i], [-1, n_input, 2250])
        output_data = np.reshape(target_data[i], [1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, prediction] ,feed_dict={x:input_data, y:output_data})
        
        loss_total += loss
        acc_total += acc
        
        
        if (i+1) % display_step == 0 :
            result = []
            strr = str(i+1)
            l = loss_total/display_step
            print("Iter= " + strr + ", Average Loss= " + \
                  "{:.6f}".format(l) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            result.append(strr)
            result.append(l)
            
            symbols_out_pred = re_dic[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (input_words[i],output_words[i],symbols_out_pred))
            
            acc_total = 0
            loss_total = 0
            
            allresult.append(result)
    
    save_path = saver.save(session, "./save_lstm1/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))