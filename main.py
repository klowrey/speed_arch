
import argparse
import sys
import tensorflow as tf
import numpy as np
import time

import ast # for eading from file
import os
import pickle

FLAGS = None

#layertypes = ['fc', 'conv', 'pool', 'loss', 'none']
#layertypes = ['fc2', 'fc4', 'fc8', 'fc16', 'none']
from common import layertypes

ntypes = len(layertypes)
nlayer = 5

tf.reset_default_graph()

# hypers
nn_train_epochs = 12 # also favors smaller networks
#nsamples = 24 # numT
nsamples = 12 # numT
total_eps = 100

#TESTING = True
TESTING = False
if TESTING:
    nn_train_epochs = 2 # also favors smaller networks
    nsamples = 12 # numT
    total_eps = 5

badnet = ['input']
badnet.extend(['none']*(nlayer))
# w_layer[ options ] ------ network ---- accuracy
# w_layer[ options ] --|             |-- speed
# w_layer[ options ] --|
# w_layer[ options ] --|
        
controller = tf.Graph()
with controller.as_default() as g:
    # inputs, action (layer choices) and reward (after training)
    actions = [ tf.placeholder(shape=[1], dtype=tf.int32) ]
    reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
    
    weights = [ tf.Variable(tf.ones([ntypes])) ]
    # need one per layer
    for i in range(nlayer-1):
        weights.append( tf.Variable(tf.ones([ntypes])) )
        actions.append( tf.placeholder(shape=[1], dtype=tf.int32) )
    
    tvars = tf.trainable_variables()
    
    #pickedactions = [ tf.argmax(w, 0) for w in weights ]
    pickedactions = [ tf.nn.softmax(w) for w in weights ]
    meanactions = [ tf.argmax(w, 0) for w in weights ]
    
    #reswght = [ tf.slice(w,a,[1]) for w, a in zip(weights, actions) ]
    reswght = [ tf.slice(w,a,[1]) for w, a in zip(pickedactions, actions) ]
    
    loglikelihood= tf.reduce_sum([ tf.log(r) for r in reswght ])
    
    #l2 = tf.reduce_sum([ tf.nn.l2_loss(w) for w in weights ])
    l2 = tf.sqrt(tf.reduce_sum([ tf.square(w) for w in weights ]))
    
    beta = tf.constant(0.6)
    loss = -(loglikelihood*reward_holder) + beta*l2
    
    getgrads = tf.gradients(loss, tvars)
    
    batchgrad = [ tf.placeholder(tf.float32) for w in weights ]
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
    #update = optimizer.minimize(loss)
    gradstep = optimizer.apply_gradients(zip(batchgrad, tvars))
    
    
nparallel = 1 # i.e. max threads
if nsamples < nparallel:
    nparallel = nsamples

choices = list(range(ntypes))
avg_reward = 0.0
reward_curve = np.zeros(total_eps, dtype=np.float32)
alpha = 0.9
acc_w = 10.0
spd_w = -100.0
prm_w = -1/10000.0
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#with tf.Session(graph = controller, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
runs = 0
with tf.Session(graph=controller, config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    gradbuffer = sess.run(tvars)
    actionlist = np.zeros(nlayer, dtype=np.int32)
    #total_reward = np.zeros(ntypes) 
    ep = 0
    while ep < total_eps:
        print('******************************************************* EPISODE:', ep)
        print('******************************************************* EPISODE:', ep)
        print('******************************************************* EPISODE:', ep)
        print('******************************************************* EPISODE:', ep)

        for ix,grad in enumerate(gradbuffer):
            gradbuffer[ix] = grad*0 # zero out the gradients

        print(ep, 'ep loop')
        reward = np.zeros(nsamples, dtype=np.float32)
        # collect gradients
        #for s in range(nsamples):
        s =0
        while s < nsamples:
            print(s, ' sample loop')
            d_files = []
            for proc in range(nparallel): # launch parallel processes
                for i in range(nlayer):
                    actionlist[i] = np.random.choice(choices, p=sess.run(pickedactions[i]))
    
                net = ['input']
                netstr = 'input'
                for i in range(nlayer):
                    net.append(layertypes[actionlist[i]])
                    netstr = netstr+' '+layertypes[actionlist[i]]

                out_file = '/tmp/m_'+str(runs)+'.txt'
                runs += 1
                d_files.append(out_file)
                #command='python3 do_mnist.py --netlist '+netstr+' --gpu 0 --out_file '+out_file+' &'
                command='python3 do_mnist.py --netlist '+netstr+' --gpu 0 --out_file '+out_file+' > /dev/null 2>&1 &'
                if TESTING:
                    fake='python3 fake_mnist.py --out_file ' + out_file+' &'
                    print(proc, fake)
                    os.system(fake) # run in background
                else:
                    print(proc, command)
                    os.system(command) # run in background
                
            # wait for parallel processes
            print(d_files)
            while all(os.path.isfile(f) for f in d_files) == False:
                #print('Waiting')
                time.sleep(2)

            #acc, speed = build_network_and_train(net)
            #reward[s] = 10.0*acc #- 100.0 * speed
            # collect results from files
            for d_file in d_files:
                with open(d_file, 'r') as f:
                    raw = f.read()
                    data = ast.literal_eval(raw)

                reward[s]  = acc_w * data['accuracy']
                #reward[s] += prm_w * data['nparams'] 
                #reward[s] += spd_w * data['speed']
                print(reward[s], 'Accuracy', data['accuracy'], 'Speed:', data['speed'])
    
                inputargs = {reward_holder:[reward[s] - avg_reward]}
                for i in range(nlayer):
                    inputargs[actions[i]] = [ actionlist[i] ]
    
                grads = sess.run(getgrads, feed_dict=inputargs)
                for ix,grad in enumerate(grads):
                    gradbuffer[ix] += grad # sum the gradients

                s += 1

        print('Reward:')
        print(reward)
        print(np.mean(reward))
    
        # gradients collected, step
        gradstepinputs = {batchgrad[nlayer-1]: gradbuffer[nlayer-1]}
        for i in range(nlayer-1):
            gradstepinputs[batchgrad[i]] = gradbuffer[i]

        avg_reward += alpha*(np.mean(reward) - avg_reward) # baseline
        reward_curve[ep] = np.mean(reward)

        sess.run(gradstep, feed_dict=gradstepinputs) # update gradients with batch

        net = ['input']
        for i in range(nlayer):
            actionlist[i] = sess.run(meanactions[i])
            net.append(layertypes[actionlist[i]])
        print(net)

        print('Weight Matrix Softmax')
        for i in range(nlayer):
            print(sess.run(pickedactions[i]))

        ep += 1

    # mean
    net = ['input']
    for i in range(nlayer):
        actionlist[i] = sess.run(meanactions[i])
        net.append(layertypes[actionlist[i]])
    print(net)

    print('Weight Matrix Softmax')
    wsm = np.zeros((nlayer, ntypes), dtype=np.float32)
    ws = np.zeros((nlayer, ntypes), dtype=np.float32)
    for i in range(nlayer):
        wsm[i,:] = sess.run(pickedactions[i])
        ws[i,:] = sess.run(weights[i])
        print(wsm[i,:])


d = {'reward': reward_curve, 'net': net, 'weights': ws, 'softmax':wsm}
pickle.dump(d, open('/tmp/params.pkl', 'wb'))

# TODO 
# keep track of average speed
# keep track of average accuracy from RL rollouts
# should be able to show they go down / stay same respectively


# with speed cost, found:
# [ input, none, none, fc2, fc4, fc2 ]
# but could be buggy

#without speed:
# input, fc2, fc16, none, fc16, fc2
# input, fc2, fc8, fc16, fc16, fc16
