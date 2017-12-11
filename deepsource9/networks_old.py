import os
from preprocessors import *
from data_prepration import *
import tensorflow as tf

class ps_detect(object):

	def __init__(self,arch,learning_rate = 0.01,n_ch = 45,hlp=7):

		self.arch = arch
		self.hlp = hlp
		self.n_ch = n_ch

		self.x_in = tf.placeholder(tf.float32,[None,2*hlp+1,2*hlp+1,n_ch])
		self.y_true = tf.placeholder(tf.float32,[None,2])

		if arch==1:
			self.x1 = tf.layers.conv2d(self.x_in,filters=3,kernel_size=1,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x2 = tf.layers.conv2d(self.x1,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x3 = tf.layers.max_pooling2d(self.x2,pool_size=2,strides=2)
			self.x4 = tf.layers.conv2d(self.x3,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x_out = tf.layers.dropout(self.x4, 0.5)

		if arch==2:
			self.x1 = tf.layers.conv2d(self.x_in,filters=3,kernel_size=1,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x2 = tf.layers.conv2d(self.x1,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x3 = tf.layers.conv2d(self.x2,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x4 = tf.layers.max_pooling2d(self.x3,pool_size=2,strides=2)
			self.x4 = tf.layers.conv2d(self.x4,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x5 = tf.layers.max_pooling2d(self.x4,pool_size=2,strides=2)
			self.x5 = tf.layers.conv2d(self.x5,filters=1,kernel_size=1,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x_out = tf.layers.dropout(self.x5, 0.5)

		if arch==3:
			self.x1 = tf.layers.conv2d(self.x_in,filters=3,kernel_size=1,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x2 = tf.layers.conv2d(self.x1,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x3 = tf.layers.conv2d(self.x2,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x4 = tf.layers.conv2d(self.x3,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x5 = tf.layers.conv2d(self.x4,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x5 = self.x5+self.x2
			self.x5 = tf.layers.batch_normalization(self.x5)

			self.x6 = tf.layers.max_pooling2d(self.x5,pool_size=2,strides=2)
			self.x6 = tf.layers.conv2d(self.x6,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x7 = tf.layers.conv2d(self.x6,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x8 = tf.layers.conv2d(self.x7,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x9 = tf.layers.conv2d(self.x8,filters=15,kernel_size=3,strides=(1, 1),padding='same',
					activation=tf.nn.relu)
			self.x9 = self.x9+self.x6
			self.x9 = tf.layers.batch_normalization(self.x9)

			self.x10 = tf.layers.max_pooling2d(self.x9,pool_size=2,strides=2)
			self.x11 = tf.layers.conv2d(self.x10,filters=1,kernel_size=1,strides=(1, 1),padding='same',
					activation=tf.nn.relu)

			self.x_out = tf.layers.dropout(self.x11, 0.5)



		self.x_out = tf.contrib.layers.flatten(self.x_out)
		self.y_pred = tf.layers.dense(self.x_out,units=2,activation=tf.nn.softmax)

		self.cost = tf.losses.softmax_cross_entropy(self.y_true,self.y_pred)
		# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


	def train(self, data_provider,restore=False,training_epochs = 1,n_sample = 1):

		init = tf.global_variables_initializer()
		sess = tf.InteractiveSession()
		saver = tf.train.Saver()

		if restore:
			saver.restore(sess, 'model'+str(self.arch)+'/model')
		else:
			sess.run(init)

		for epoch in range(training_epochs):
				# Loop over all batches
				cc = 0
				ii = 0
				for i in range(10):
				    while True:
				        xp,yp = data_provider(1)
				        xp = preprocess(xp)    
				        xb,yb = slide_batch(xp,yp,n_sample=n_sample,hlp=7)
				        if xb is not None:
				            break
				    # Run optimization op (backprop) and cost op (to get loss value)
				    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x_in: xb, self.y_true: yb})
				    cc += c
				    ii += 1
				# Display logs per epoch step
				print 'Epoch:{:d}, cost= {:f}'.format(epoch, cc/ii)

		# Creates a saver.
		ch_mkdir('model'+str(self.arch))
		saver.save(sess, 'model'+str(self.arch)+'/model')

	def pre_eval(self, data_provider,n_try = 20,n_sample=20):

		init = tf.global_variables_initializer()
		sess = tf.InteractiveSession()
		new_saver = tf.train.Saver()
		new_saver.restore(sess, './model'+str(self.arch)+'/model')

		acu = np.zeros(n_try)

		for i in range(n_try):
			while True:
				xp,yp = data_provider(1)
				xp = preprocess(xp)    
				xb,yb = slide_batch(xp,yp,n_sample=n_sample,hlp=7)
				if xb is not None:
					break
			# Run optimization op (backprop) and cost op (to get loss value)
			yt_pred = sess.run(self.y_pred, feed_dict={self.x_in: xb})
			acu[i] = (np.argmax(yt_pred,axis=1)==np.argmax(yb,axis=1)).sum()/(3.*n_sample)

		print acu.mean(),acu.std()

	def slide_proba(self,xs):

		hlp = self.hlp
		n_ch = self.n_ch
		init = tf.global_variables_initializer()
		sess = tf.InteractiveSession()
		new_saver = tf.train.Saver()
		new_saver.restore(sess, './model'+str(self.arch)+'/model')

		lx,ly = xs.shape[1:3]
		x_s = np.zeros((lx-2*hlp,2*hlp+1,2*hlp+1,n_ch))
		x_p = np.zeros((lx,ly))
		for j in range(ly-2*hlp):
				for i in range(lx-2*hlp):
				    x_s[i,:,:,:] = xs[0,i:i+2*hlp+1,j:j+2*hlp+1,:]
				    
				yt_pred = sess.run(self.y_pred, feed_dict={self.x_in: x_s})
				yt_pred = yt_pred[:,1]
				x_p[hlp:lx-hlp,j+hlp] = yt_pred

		return x_p

	def conv_layer(self,xs,step = None):

		if step is None:
			step = self.hlp
		hlp = self.hlp
		n_ch = self.n_ch
		s_left = hlp-step//2-1
		s_right = hlp+step//2+1
		init = tf.global_variables_initializer()
		sess = tf.InteractiveSession()
		new_saver = tf.train.Saver()
		new_saver.restore(sess, './model'+str(self.arch)+'/model')

		lx,ly = xs.shape[1:3]
		x_s = np.zeros((lx-2*hlp,2*hlp+1,2*hlp+1,n_ch))
		x_c = np.zeros((lx,ly,3))

		for j in range(0,ly-2*hlp,step):
				for i in range(0,lx-2*hlp,step):
				    x_s[i,:,:,:] = xs[0,i:i+2*hlp+1,j:j+2*hlp+1,:]
				x_sc = sess.run(self.x1, feed_dict={self.x_in: x_s})

				for i in range(0,lx-2*hlp,step):
				    x_c[i+s_left:i+s_right,j+s_left:j+s_right,:] =x_sc[i,s_left:s_right,s_left:s_right,:]

		return x_c

def ch_mkdir(directory):
	if not os.path.exists(directory):
		  os.makedirs(directory)
