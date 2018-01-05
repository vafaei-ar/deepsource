import os
import sys
import numpy as np
import tensorflow as tf
from util import ch_mkdir
from sw_provider import slide_batch
from preprocessors import preprocess

class ConvolutionalLayers(object):
	"""
	CLASS ConvolutionalLayers: This class provides you to define, train, restore and operate a sequential convolutional neural network.
	
	--------
	METHODS:
	
	__init__:
	|	Arguments:
	| 	nl (default=400): input layer size.
	| 	learning_rate (default=0.001): learning rate.
	| 	n_channel (default=1): number of channels in input layer.
	| 	restore (default=False): restore flag.
	| 	model_add (default='./model'): saved model address / the address trained model will be saved.
	| 	arch_file_name (default=None): name of the architecture file. This file should be designed similar to the sample and located in same directory with the script.
	
	restore:
	This method restores a saved model with specific architecture.
	|	Arguments:
	| 	No argument. 
	
	|	Returns:
	|		null
	
	train:
	This method trains CNN.
	|	Arguments:
	| 	data_provider: data provider class to feed CNN.
	|		training_epochs (default=1): number of training epochs.
	|		n_s (default=1): number of used image(s) in each epoch.
	|		dropout (default=0.5): dropout.
	|		time_limit (default=None): time limit of training in minutes.
	
	|	Returns:
	|		null
	
	conv:
	This method convolve an image using trained network.
	|	Arguments:
	|		x_in: input image.
	
	| Returns:
	|		2D convolved image.
	
	conve large image:
	This method convolve a large image using trained network.
	|	Arguments:
	|		xsm: input image.
	|		pad (default=10): shared pad between windows.
	|		lw (default=400):	window size.
	
	| Returns:
	|		2D convolved image.
	
	"""
	def __init__(self,nl=400,learning_rate = 0.001,n_channel=1,restore=False,model_add='./model',arch_file_name=None):

		self.model_add = model_add
		self.x_in = tf.placeholder(tf.float32,[None,nl,nl,n_channel])
		self.y_true = tf.placeholder(tf.float32,[None,nl,nl,1])
		self.drop_out = tf.placeholder(tf.float32)
		self.nl = nl

		if arch_file_name is not None:
			if arch_file_name[-3:]=='.py':
				arch_file_name = arch_file_name[-3:]
			exec 'import '+arch_file_name+' as arch'
			self.outputs = arch.architecture(self.x_in, self.drop_out)
			if type(self.outputs) is list:
				self.x_out = self.outputs[0]
			else:
				self.x_out = self.outputs
		else:

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

		self.cost = tf.reduce_sum(tf.pow(self.y_true - self.x_out, 2))
#		self.cost = tf.losses.log_loss(self.y_true,self.x_out)

		self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
#		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver()

		if restore:
			tf.reset_default_graph()
			self.saver.restore(self.sess, model_add+'/model')
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def restore(self):
		tf.reset_default_graph()
		self.saver.restore(self.sess, self.model_add+'/model')

	def train(self, data_provider,training_epochs = 1,n_s = 1, dropout=0.5, time_limit=None):

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
				    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x_in: xb, self.y_true: yb, self.drop_out: dropout})
				    cc += c
				    ii += 1
				# Display logs per epoch step
				print 'Epoch:{:d}, cost= {:f}'.format(epoch, cc/ii)
				if time_limit is not None:
					t1 = time.time()
					if (t1-t0)/60>time_limit:
						print "Time's up, goodbye!"
						ch_mkdir(self.model_add)
						self.saver.save(self.sess, self.model_add+'/model')
						return 0

		# Creates a saver.
		ch_mkdir(self.model_add)
		self.saver.save(self.sess, self.model_add+'/model')

	def conv(self,x_in):
		x_out = self.sess.run(self.x_out, feed_dict={self.x_in: x_in, self.drop_out: 1.})
		return x_out[0,:,:,0]

	def conv_large_image(self,xsm,pad=10,lw=400):
		prm = np.zeros((xsm.shape[1],xsm.shape[2]))
		li = xsm.shape[1]
		for i in np.arange(0,li,lw-2*pad):
			if i+lw<li:
				iii = i
				iei = i+lw
			else:
				iii = li-lw
				iei = li

			for j in np.arange(0,li,lw-2*pad):
				if j+lw<li:
				  jii = j
				  jei = j+lw
				else:
				  jii = li-lw
				  jei = li

				conx = self.conv(xsm[:,iii:iei,jii:jei,:])
				prm[iii+pad:iei-pad,jii+pad:jei-pad] = conx[pad:-pad,pad:-pad]
		return prm

###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################
###############################################

class SW_Network(object):
	"""
	CLASS SW_Network: This class provides you to define, train, restore and operate a sliding convolutional neural network. (This class in under construction!!!)
	
	--------
	METHODS:
	
	__init__:
	|	Arguments:
	| 	nl (default=400): input layer size.
	| 	learning_rate (default=0.001): learning rate.
	| 	n_channel (default=1): number of channels in input layer.
	| 	restore (default=False): restore flag.
	| 	model_add (default='./model'): saved model address / the address trained model will be saved.
	| 	arch_file_name (default=None): name of the architecture file. This file should be designed similar to the sample and located in same directory with the script.
	
	restore:
	This method restores a saved model with specific architecture.
	|	Arguments:
	| 	No argument. 
	
	|	Returns:
	|		null
	
	train:
	This method trains CNN.
	|	Arguments:
	| 	data_provider: data provider class to feed CNN.
	|		training_epochs (default=1): number of training epochs.
	|		n_s (default=1): number of used image(s) in each epoch.
	|		dropout (default=0.5): dropout.
	|		time_limit (default=None): time limit of training in minutes.
	
	|	Returns:
	|		null
	"""
	def __init__(self,arch,learning_rate = 0.01,n_ch = 46,hlp=15,pp_conv=True,restore=False,model_add='./model',conv_add=''):

		self.arch = arch
		self.hlp = hlp
		self.n_ch = n_ch
		self.model_add = model_add
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
		if pp_conv:
			self.pp_conv = PreProcess_Conv(nl=400,learning_rate = 0.001,restore=1,model_add=conv_add)

		self.x_in = tf.placeholder(tf.float32,[None,2*hlp+1,2*hlp+1,n_ch])
		self.y_true = tf.placeholder(tf.float32,[None,2])
#		self.y_true = tf.placeholder(tf.float32,[None,1])
		self.drop_out = tf.placeholder(tf.float32)

#		self.xnn = tf.layers.conv2d(self.x_in,filters=3,kernel_size=1,strides=(1, 1),padding='same',
#				activation=tf.nn.relu,kernel_regularizer=regularizer)

		if arch==1:

#			x_in = tf.placeholder(tf.float32, shape=[None, 41,41,21])
#			self.y_true = tf.placeholder(tf.float32, shape=[None, 2])


			x = tf.layers.conv2d(self.x_in,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						      activation=tf.nn.relu)

			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)

			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						  activation=tf.nn.relu)

			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)

#			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#					activation=tf.nn.relu)

#			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
#			x = tf.layers.batch_normalization(x)
#			x = tf.layers.dropout(x, 0.5)

#			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#					activation=tf.nn.relu)

#			self.x_out = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
#			x = tf.layers.batch_normalization(x)
#			self.x_out = tf.layers.dropout(x, 0.5)

		elif arch==2:

			x = tf.layers.conv2d(self.x_in,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						      activation=tf.nn.relu)
			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						      activation=tf.nn.relu)

			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
			x = tf.layers.batch_normalization(x)

			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						  activation=tf.nn.relu)
			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						  activation=tf.nn.relu)

			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
			x = tf.layers.batch_normalization(x)

		elif arch==3:

			x = tf.layers.conv2d(self.x_in,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						      activation=tf.nn.relu)
			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						      activation=tf.nn.relu)

			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
			x = tf.layers.batch_normalization(x)

			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						  activation=tf.nn.relu)
			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
						  activation=tf.nn.relu)

			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
			x = tf.layers.batch_normalization(x)

#			xp = tf.layers.conv2d(self.x_in,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#						      activation=tf.nn.relu)
#			x = tf.layers.conv2d(xp,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#						      activation=tf.nn.relu)
#			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#						      activation=tf.nn.relu)

#			x = x+xp

#			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
#			xp = tf.layers.batch_normalization(x)

#			x = tf.layers.conv2d(xp,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#						  activation=tf.nn.relu)
#			x = tf.layers.conv2d(x,filters=16,kernel_size=3,strides=(1, 1),padding='same',
#						  activation=tf.nn.relu)

#			x = x+xp
#			x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)
#			x = tf.layers.batch_normalization(x)

		else:
			assert False, 'Unknown architecture!'

#		x = tf.layers.conv2d(self.x_out,filters=1,kernel_size=3,strides=(1, 1),padding='same',
#				activation=tf.nn.relu)
#		x = tf.layers.max_pooling2d(x,pool_size=2,strides=2)

#		x = tf.layers.batch_normalization(x)
#		x = tf.layers.dropout(x, 0.5)

#		x = tf.contrib.layers.flatten(x)
#		self.y_pred = tf.layers.dense(x,units=2,activation=tf.nn.relu)

#		self.y_pred = tf.layers.dense(x,units=2)

#		print self.y_pred

#		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.y_pred))
#		self.cost = tf.losses.softmax_cross_entropy(self.y_true,self.y_pred)
#		self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_true * tf.log(self.y_pred), reduction_indices=[1]))
#		self.cost = tf.losses.log_loss(self.y_true,self.y_pred)

#		self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
#		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
#		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)


		x = tf.contrib.layers.flatten(x)
		x = tf.nn.dropout(x,self.drop_out)
		self.y_pred = tf.layers.dense(x,2)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.y_pred))

		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

		correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver()

		if restore:
			tf.reset_default_graph()
			self.saver.restore(self.sess, model_add+'/model')
		else:
			init = tf.global_variables_initializer()
			self.sess.run(init)

	def restore(self):
		self.saver.restore(self.sess, self.model_add+'/model')

	def train(self, data_provider,training_epochs = 1,n_s = 1, ):

		for epoch in range(training_epochs):
				# Loop over all batches
				cc = 0
				ii = 0
				while True:
						xp,yp = data_provider(1)
						xp = preprocess(xp,funcs=[self.pp_conv.pp_conv])    
						xb,yb = slide_batch(xp,yp,n_ps=n_s,n_vs=n_s,n_bg=2*n_s,hlp=20)
						if xb is not None:
								break
				# Run optimization op (backprop) and cost op (to get loss value)
				#				    _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x_in: xb, self.y_true: yb})
				#				    cc += c
				#				    ii += 1
				#				# Display logs per epoch step
				#				print 'Epoch:{:d}, cost= {:f}'.format(epoch, cc/ii)

				self.train_step.run(feed_dict={self.x_in: xb, self.y_true: yb, self.drop_out: 0.5})

				if epoch % 5 == 0:
					train_accuracy = self.accuracy.eval(feed_dict={self.x_in: xb, self.y_true: yb, self.drop_out: 1.0})
					sys.stdout.write('\rstep %d, training accuracy %g5' % (epoch, train_accuracy))
					sys.stdout.flush()

		# Creates a saver.
		ch_mkdir(self.model_add)
		self.saver.save(self.sess, self.model_add+'/model')

	def pre_eval(self, data_provider,n_try = 20,n_s=20):

		acu_ps = np.zeros(n_try)
		acu_vs = np.zeros(n_try)
		acu_bg = np.zeros(n_try)

		for i in range(n_try):
			while True:
				xp,yp = data_provider(1)
				xp = preprocess(xp,funcs=[self.pp_conv.pp_conv])    
				xb,yb = slide_batch(xp,yp,hlp=20,n_ps=n_s,n_vs=n_s,n_bg=n_s,shuffle=False)
				if xb is not None:
					break
			# Run optimization op (backprop) and cost op (to get loss value)
			acu_ps[i] = self.accuracy.eval(feed_dict={self.x_in: xb[:n_s], self.y_true: yb[:n_s], self.drop_out: 1.0})

			acu_vs[i] = self.accuracy.eval(feed_dict={self.x_in: xb[n_s:2*n_s], self.y_true: yb[n_s:2*n_s], self.drop_out: 1.0})

			acu_bg[i] = self.accuracy.eval(feed_dict={self.x_in: xb[2*n_s:3*n_s], self.y_true: yb[2*n_s:3*n_s], self.drop_out: 1.0})

		print 'Point source: m={:4.2f}, p5={:4.2f}, p95={:4.2f}'.format(acu_ps.mean(),np.percentile(acu_ps,5),np.percentile(acu_ps,95))
		print 'Vicinity: m={:4.2f}, p5={:4.2f}, p95={:4.2f}'.format(acu_vs.mean(),np.percentile(acu_vs,5),np.percentile(acu_vs,95))
		print 'Background: m={:4.2f}, p5={:4.2f}, p95={:4.2f}'.format(acu_bg.mean(),np.percentile(acu_bg,5),np.percentile(acu_bg,95))

	def slide_proba(self,xs,step=1):

		hlp = self.hlp
		n_ch = self.n_ch
		lx,ly = xs.shape[1:3]
		x_s = np.zeros((lx-2*hlp,2*hlp+1,2*hlp+1,n_ch))
		x_p = np.zeros((lx,ly))

		xp = preprocess(xs,funcs=[self.pp_conv.pp_conv])  
		for j in range(0,ly-2*hlp,step):
				for i in range(0,lx-2*hlp,step):
				    x_s[i,:,:,:] = xp[0,i:i+2*hlp+1,j:j+2*hlp+1,:]
				    
				yt_pred = self.sess.run(self.y_pred, feed_dict={self.x_in: x_s, self.drop_out: 1.0})
				yt_pred = yt_pred[:,1]
				x_p[hlp:lx-hlp:step,j+hlp] = yt_pred

		return x_p

	def conv_layer(self,xs,step = None):

		if step is None:
			step = self.hlp
		hlp = self.hlp
		n_ch = self.n_ch
		s_left = hlp-step//2-1
		s_right = hlp+step//2+1

		lx,ly = xs.shape[1:3]
		x_s = np.zeros((lx-2*hlp,2*hlp+1,2*hlp+1,n_ch))
		x_c = np.zeros((lx,ly,3))

		for j in range(0,ly-2*hlp,step):
				for i in range(0,lx-2*hlp,step):
				    x_s[i,:,:,:] = xs[0,i:i+2*hlp+1,j:j+2*hlp+1,:]
				x_sc = self.sess.run(self.xnn, feed_dict={self.x_in: x_s})

				for i in range(0,lx-2*hlp,step):
				    x_c[i+s_left:i+s_right,j+s_left:j+s_right,:] =x_sc[i,s_left:s_right,s_left:s_right,:]

		return x_c





