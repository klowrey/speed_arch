import argparse

from common import layertypes

import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--netlist', nargs='*')
parser.add_argument('--out_file')
parser.add_argument('--gpu')
args = parser.parse_args()

netlist = args.netlist
out_file = args.out_file
gpu_idx = int(args.gpu)

nlayer = len(netlist)
badnet = ['input']
badnet.extend(['none']*(nlayer-1))
def save_badnet():
    data = {}
    data['model'] = netlist
    data['accuracy'] = -1000.0
    data['speed'] = 1000.0 
    data['nparams'] = 10000000 
    
    with open(out_file, 'w') as f:
        f.write(str(data))
    sys.exit(0)

if netlist == badnet:
    save_badnet()

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, GlobalAveragePooling2D, \
        BatchNormalization, Conv2D, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam #, RMSprop 
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from depthwise_conv2d import DepthwiseConvolution2D

batch_size = 128
nn_train_epochs = 10
num_classes = 10

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    in_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    in_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)


with tf.device('/gpu:'+str(gpu_idx)):
    print(netlist)
    epochs = nn_train_epochs
    model = Sequential()
    start_idx = 1
    alpha = 1
    conv_val = 32 #64

    while netlist[start_idx] == 'none':
        start_idx += 1

    lastlayer = 'dense'
    if netlist[start_idx] == layertypes[0]:
        print("Input", layertypes[0])
        model.add(Dense(units=32, input_shape=in_shape))
        lastlayer = 'dense'
    #elif netlist[start_idx] == layertypes[1]:
    #    print("Input", layertypes[1])
    #    model.add(Dense(units=64, input_shape=in_shape))
    #    lastlayer = 'dense'
    elif netlist[start_idx] == layertypes[1]:
        print("Input", layertypes[1])
        model.add(DepthwiseConvolution2D(int(conv_val * alpha), (3, 3),
            strides=(1, 1), padding='same', use_bias=False, input_shape=in_shape))
        model.add(BatchNormalization())
        lastlayer = 'conv'
    elif netlist[start_idx] == layertypes[2]:
        print("Input", layertypes[2])
        model.add(Conv2D(int(conv_val * alpha), (3, 3),
            strides=(2, 2), padding='same', use_bias=False, input_shape=in_shape))
        model.add(BatchNormalization())
        lastlayer = 'conv'
    elif netlist[start_idx] == layertypes[3]:
        print("Input", layertypes[3])
        model.add(DepthwiseConvolution2D(int(conv_val * alpha), (3, 3),
            strides=(2, 2), padding='same', use_bias=False, input_shape=in_shape))
        model.add(BatchNormalization())
        lastlayer = 'conv'

    model.add(Activation('relu')) # just use relu for now

    for i in range(start_idx+1,nlayer):
        layertype = netlist[i]

        if layertype == layertypes[0]:
            print("Adding", layertypes[0])
            model.add(Dense(units=32))
            model.add(Activation('relu')) # just use relu for now
            lastlayer = 'dense'
        #elif layertype == layertypes[1]:
        #    print("Adding", layertypes[1])
        #    model.add(Dense(units=64))
        #    model.add(Activation('relu')) # just use relu for now
        #    lastlayer = 'dense'
        elif layertype == layertypes[1]:
            print("Adding", layertypes[1])
            model.add(DepthwiseConvolution2D(int(conv_val * alpha), (3, 3),
                strides=(1, 1), padding='same', use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu')) # just use relu for now
            lastlayer = 'conv'
        elif layertype == layertypes[2]:
            print("Adding", layertypes[2])
            model.add(Conv2D(int(conv_val * alpha), (3, 3),
                strides=(2, 2), padding='same', use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu')) # just use relu for now
            lastlayer = 'conv'
        elif layertype == layertypes[3]:
            print("Adding", layertypes[4])
            model.add(DepthwiseConvolution2D(int(conv_val * alpha), (3, 3),
                strides=(2, 2), padding='same', use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu')) # just use relu for now
            lastlayer = 'conv'
        #model.summary()

    if lastlayer == 'dense':
        print('Flattening Dense')
        model.add(Flatten()) # needed with conv layers
    else:
        print('Global Pool')
        model.add(GlobalAveragePooling2D())
    model.add(Dense(units=num_classes)) # final layer
    model.add(Activation('softmax'))

    # done building model
    #model.summary()
    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Loss', score[0], 'Accuracy', score[1])

    avg = 0
    runs = 10
    for i in range(runs):
        start = time.time()
        model.predict(x_test, batch_size=batch_size, verbose=0)
        end = time.time()
        avg += (end-start)
    speed = avg / runs
    print('GPU Speed', speed)

nparams = model.count_params()
with tf.device('/cpu:0'):
    cpu_time = model
    avg = 0
    runs = 10
    for i in range(runs):
        start = time.time()
        cpu_time.predict(x_test, batch_size=batch_size, verbose=0)
        end = time.time()
        avg += (end-start)
    speed = avg / runs
    print('CPU Speed', speed)

#K.clear_session()
#tf.reset_default_graph()

print(score[1], speed, nparams)

data = {}
data['model'] = netlist
data['accuracy'] = score[1]
data['speed'] = speed 
data['nparams'] = nparams 

with open(out_file, 'w') as f:
    f.write(str(data))

