import tensorflow as tf

def architecture(x_in,drop_out):

	xp = tf.layers.conv2d(x_in,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x = tf.layers.conv2d(xp,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x = tf.layers.conv2d(x,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)
	x = x+xp
	x = tf.layers.batch_normalization(x)
	x = tf.layers.conv2d(x,filters=12,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	x = tf.layers.dropout(x, drop_out)
	x_out = tf.layers.conv2d(x,filters=1,kernel_size=5,strides=(1, 1),padding='same',
			activation=tf.nn.relu)

	return x_out
