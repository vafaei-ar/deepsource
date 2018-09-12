from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from .utils import ch_mkdir,the_print

class ConvolutionalLayers(object):
    """
    CLASS ConvolutionalLayers: This class provides you to define, train, restore and operate a sequential convolutional neural network.
    
    --------
    METHODS:
    
    __init__:
    |    Arguments:
    |     nl (default=400): input layer size.
    |     learning_rate (default=0.001): learning rate.
    |     n_channel (default=1): number of channels in input layer.
    |     restore (default=False): restore flag.
    |     model_add (default='./model'): saved model address / the address trained model will be saved.
    |     arch_file_name (default=None): name of the architecture file. This file should be designed similar to the sample and located in same directory with the script.
    
    restore:
    This method restores a saved model with specific architecture.
    |    Arguments:
    |     No argument. 
    
    |    Returns:
    |        null
    
    train:
    This method trains CNN.
    |    Arguments:
    |     data_provider: data provider class to feed CNN.
    |        training_epochs (default=1): number of training epochs.
    |        n_s (default=1): number of used image(s) in each epoch.
    |        dropout (default=0.5): dropout.
    |        time_limit (default=None): time limit of training in minutes.
    
    |    Returns:
    |        null
    
    conv:
    This method convolve an image using trained network.
    |    Arguments:
    |        x_in: input image.
    
    | Returns:
    |        2D convolved image.
    
    conve large image:
    This method convolve a large image using trained network.
    |    Arguments:
    |        xsm: input image.
    |        pad (default=10): shared pad between windows.
    |        lw (default=400):    window size.
    
    | Returns:
    |        2D convolved image.
    
    """
    def __init__(self,nx=200,ny=None,n_channel=1,restore=False,model_add='./model',arch_file_name=None):

        tf.reset_default_graph()
        self.model_add = model_add
        if ny is None:
            ny = nx
        self.x_in = tf.placeholder(tf.float32,[None,nx,ny,n_channel])
        self.y_true = tf.placeholder(tf.float32,[None,nx,ny,1])
        self.learning_rate = tf.placeholder(tf.float32)
        self.drop_out = tf.placeholder(tf.float32)
        self.nx = nx
        self.ny = ny

        if arch_file_name is not None:
            if arch_file_name[-3:]=='.py':
                arch_file_name = arch_file_name[-3:]
            exec('from '+arch_file_name+' import architecture', globals())
            self.outputs = architecture(self.x_in, self.drop_out)
            try:
                os.remove(arch_file_name+'.pyc')
            except:
                pass
            try:
                shutil.rmtree('__pycache__')
            except:
                pass

            if type(self.outputs) is list:
                self.x_out = self.outputs[0]
            else:
                self.x_out = self.outputs
        else:
            self.outputs = self.architecture()

        self.cost = tf.reduce_sum(tf.pow(self.y_true - self.x_out, 2))
#        self.cost = tf.losses.log_loss(self.y_true,self.x_out)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
#        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()

        if restore:
            self.saver.restore(self.sess, model_add+'/model')
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
                
    def architecture(self):
        xp = tf.layers.conv2d(self.x_in,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)
        x = tf.layers.conv2d(xp,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)
        x = tf.layers.conv2d(x,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)

        x = x+xp
        x = tf.layers.batch_normalization(x)

        x = tf.layers.conv2d(x,filters=16,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)

        x = tf.layers.dropout(x, self.drop_out)
        self.x_out = tf.layers.conv2d(x,filters=1,kernel_size=5,strides=(1, 1),padding='same',
                activation=tf.nn.relu)
        return self.x_out

    def restore(self):
        tf.reset_default_graph()
        self.saver.restore(self.sess, self.model_add+'/model')

    def train(self, data_provider,training_epochs = 1,n_s = 1, learning_rate = 0.001, dropout=0.5, time_limit=None, verbose=0):

        if time_limit is not None:
            import time
            t0 = time.time()

        for epoch in range(training_epochs):
                # Loop over all batches
                cc = 0
                ii = 0
                for i in range(10):
                    while True:
                        xb,yb = data_provider(n_s)
                        if xb is not None:
                            break
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x_in: xb, self.y_true: yb, self.drop_out: dropout, self.learning_rate: learning_rate})
                    cc += c
                    ii += 1
                # Display logs per epoch step
                if verbose:
                    if epoch%verbose==0:
                        print('Epoch:{:d}, cost= {:f}'.format(epoch, cc/ii))
                if time_limit is not None:
                    t1 = time.time()
                    if (t1-t0)/60>time_limit:
                        the_print("Time's up, goodbye!",tc='red',bgc='green')
                        ch_mkdir(self.model_add)
                        self.saver.save(self.sess, self.model_add+'/model')
                        return 0

        # Creates a saver.
        ch_mkdir(self.model_add)
        self.saver.save(self.sess, self.model_add+'/model')

    def conv(self,x_in):
        x_out = self.sess.run(self.x_out, feed_dict={self.x_in: x_in, self.drop_out: 1.})
        return x_out[0,:,:,0]

    def conv_large_image(self,xsm,pad=10,lx=200,ly=None):
        if ly is None:
            ly = lx
        prm = np.zeros((xsm.shape[1],xsm.shape[2]))
        li = xsm.shape[1]
        lj = xsm.shape[2]
        for i in np.arange(0,li,lx-2*pad):
            if i+lx<li:
                iii = i
                iei = i+lx
            else:
                iii = li-lx
                iei = li

            for j in np.arange(0,lj,ly-2*pad):
                if j+ly<lj:
                  jii = j
                  jei = j+ly
                else:
                  jii = lj-ly
                  jei = lj

                conx = self.conv(xsm[:,iii:iei,jii:jei,:])
                prm[iii+pad:iei-pad,jii+pad:jei-pad] = conx[pad:-pad,pad:-pad]
        return prm

    def get_filters(self):
        filts = [str(i.name).split('/')[0] for i in tf.trainable_variables() if 'kernel' in i.name]    
        weights = []
        for filt in filts:
            with tf.variable_scope(filt, reuse=True) as scope_conv:
                W_conv = tf.get_variable('kernel')
                weights.append(W_conv.eval())
                
        return weights
                
                
