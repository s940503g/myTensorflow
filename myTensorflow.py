#import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class myDenseNet:
	def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None, learning_rate=0.0001):
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.learning = learning_rate
		
		return None	
	def create_model(self):
		input_layer = self.create_conv_layer(self.x_shaped, 1, self.k_0, [7, 7])
		input_layer = self.create_max_pooling(input_layer, [3, 3])
		print("input_layer: ", input_layer.get_shape());

		dense_block_1 = self.create_dense_block(input_layer, layers=1)
		trans_layer_1 = self.create_transition_layer(dense_block_1)

		dense_block_2 = self.create_dense_block(trans_layer_1, layers=2)
		trans_layer_2 = self.create_transition_layer(dense_block_2)

		dense_block_3 = self.create_dense_block(trans_layer_2, layers=3)
		trans_layer_3 = self.create_transition_layer(dense_block_3)

		dense_block_4 = self.create_dense_block(trans_layer_3, layers=4)
		trans_layer = self.create_transition_layer(dense_block_4)

		avg_pooling_layer = self.create_avg_pooling(trans_layer, [7, 7])
		dense_layer = self.create_dense_layer(avg_pooling_layer, [1000, 10])

		y_ = tf.nn.softmax(dense_layer)

		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=self.y))

		# add an optimiser
		optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

		# define an accuracy assessment operation
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# setup the initialisation operator
		init_op = tf.global_variables_initializer()

		gpu_options = tf.GPUOptions(allow_growth=True)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
			allow_soft_placement=True)) as sess:
			# initialise the variables
			sess.run(init_op)
			total_batch = int(len(self.y) / self.batch_size)
			for epoch in range(self.epochs):
				avg_cost = 0
				train_acc = 0
				test_acc = 0

				for i in range(total_batch):
					batch_x, batch_y = self.next_batch(x=self.train_x, y=self.train_y, batch_size=self.batch_size)
					_, c = sess.run([optimiser, cross_entropy],feed_dict={self.x: batch_x, self.y: batch_y})
					avg_cost += c / total_batch
				
				test_acc = sess.run(accuracy, feed_dict={self.x:self.test_x, self.y: self.test_y})
				print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "testing accuracy: {:.4f}".format(test_acc))
				

			print("\nTraining complete!")
			#test_acc = sess.run(accuracy, feed_dict={self.x:self.mnist.test.images, self.y: self.mnist.test.labels})
			#print('Testing accuracy: {:.4f}'.format(test_acc))
		return 0

	def next_batch(x,y, batch_size=50):
		idx = np.arange(0, len(data))
		np.random.shuffle(idx)
		idx = idx[:batch_size]
		data_shuffle = [data[i] for i in idx]
		labels_shuffle = [labels[i] for i in idx]

		return np.asarray(data_shuffle), np.asarray(labels_shuffle)
	def create_conv_layer(self, input_data, num_input_channels, num_filters, filter_shape, dropout_rate=0.0):
		conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
		weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03))
		out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

		# add bias
		bias = tf.Variable(tf.truncated_normal([num_filters]))
		out_layer = tf.nn.bias_add(out_layer, bias)

		# apply a ReLU dropout
		if dropout_rate:
			out_layer = tf.nn.dropout(out_layer, 1.0-dropout_rate)
		# apply a ReLU non-linear activation
		return out_layer

	def create_relu_layer(self, input_layer):
		out_layer = tf.nn.relu(input_layer)
		return out_layer

	def create_batch_layer(self, input_layer):
		# Batch normalization
		input_layer_shape = input_layer.get_shape().as_list()
		batch_mean, batch_var = tf.nn.moments(input_layer, [0])
		scale = tf.Variable(tf.ones([input_layer_shape[1], input_layer_shape[2],input_layer_shape[3]]))
		beta = tf.Variable(tf.zeros([input_layer_shape[1], input_layer_shape[2],input_layer_shape[3]]))
		epsilon = 1e-3
		out_layer = tf.nn.batch_normalization(input_layer, batch_mean, batch_var, beta, scale, epsilon)
		return out_layer

	def create_max_pooling(self, input_layer, pool_shape):
		ksize = [1, pool_shape[0], pool_shape[1], 1]
		strides = [1, 2, 2, 1]
		out_layer = tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding='SAME')
		return out_layer

	def create_avg_pooling(self, input_layer, pool_shape):
		ksize = [1, pool_shape[0], pool_shape[1], 1]
		strides = [1, 2, 2, 1]
		out_layer = tf.nn.avg_pool(input_layer, ksize=ksize, strides=strides, padding='SAME')
		return out_layer

	def create_flat_layer(self, input_layer):
		input_layer_shape = input_layer.get_shape().as_list()
		input_nodes = input_layer_shape[1] * input_layer_shape[2] * input_layer_shape[3]
		flattened = tf.reshape(input_layer, [-1, input_nodes])
		return flattened

	def create_dense_layer(self, input_layer, hidden_layer):
		flattened = self.create_flat_layer(input_layer)
		input_nodes = flattened.get_shape().as_list()[1]
		wd1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer[0]], stddev=0.03))
		bd1 = tf.Variable(tf.truncated_normal([hidden_layer[0]], stddev=0.01))

		dense_layer1 = tf.matmul(flattened, wd1) + bd1
		dense_layer1 = tf.nn.relu(dense_layer1)

		wd2 = tf.Variable(tf.truncated_normal(hidden_layer, stddev=0.03), name='wd2')
		bd2 = tf.Variable(tf.truncated_normal([hidden_layer[1]], stddev=0.01), name='bd2')
		dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
		return dense_layer2

	def create_concat_layer(self, input_layers):
		out_layer = tf.concat(values=input_layers, axis=3)
		return out_layer
	
	def create_dense_block(self, input_layer, layers):
		shape = input_layer.get_shape().as_list()
		maps = self.k_0 + (layers - 1) * self.k
		conv_1x1 = self.create_batch_layer(input_layer)
		conv_1x1 = self.create_relu_layer(conv_1x1)
		conv_1x1 = self.create_conv_layer(input_layer, shape[3], maps, [1, 1])

		shape = conv_1x1.get_shape().as_list()
		conv_3x3 = self.create_batch_layer(conv_1x1)
		conv_3x3 = self.create_relu_layer(conv_3x3)
		conv_3x3 = self.create_conv_layer(conv_3x3, shape[3], maps, [3, 3])
		
		shape = conv_3x3.get_shape().as_list()
		conv_7x7 = self.create_batch_layer(conv_3x3)
		conv_7x7 = self.create_relu_layer(conv_7x7)
		conv_7x7 = self.create_conv_layer(conv_7x7, shape[3], maps, [7, 7])
		
		out_layer = self.create_concat_layer([conv_1x1, conv_3x3, conv_7x7])
		out_layer = self.create_batch_layer(out_layer)
		print('create dense block:', out_layer.get_shape())
		layers += 1
		return out_layer

	def create_transition_layer(self, input_layer):	
		shape = input_layer.get_shape().as_list()

		out_layer = self.create_batch_layer(input_layer)
		out_layer = self.create_relu_layer(out_layer)
		out_layer = self.create_conv_layer(out_layer, shape[3], shape[3], [1, 1])
		
		out_layer = self.create_max_pooling(out_layer, [2, 2])

		print('transition layer:', out_layer.get_shape())

		return out_layer

	def test_by_mnist(self):
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		# Python optimisation variables
		self.learning_rate = 0.0001
		self.epochs = 50
		self.batch_size = 50
		self.k_0 = 32
		self.k = 4
		
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.x_shaped = tf.reshape(self.x, [-1, 28, 28, 1])
		self.y = tf.placeholder(tf.float32, [None, 10])		
	
		input_layer = self.create_conv_layer(self.x_shaped, 1, self.k_0, [7, 7])
		input_layer = self.create_max_pooling(input_layer, [3, 3])
		print("input_layer: ", input_layer.get_shape());

		dense_block_1 = self.create_dense_block(input_layer, layers=1)
		trans_layer_1 = self.create_transition_layer(dense_block_1)

		dense_block_2 = self.create_dense_block(trans_layer_1, layers=2)
		trans_layer_2 = self.create_transition_layer(dense_block_2)

		dense_block_3 = self.create_dense_block(trans_layer_2, layers=3)
		trans_layer_3 = self.create_transition_layer(dense_block_3)

		dense_block_4 = self.create_dense_block(trans_layer_3, layers=4)
		trans_layer = self.create_transition_layer(dense_block_4)

		avg_pooling_layer = self.create_avg_pooling(trans_layer, [7, 7])
		dense_layer = self.create_dense_layer(avg_pooling_layer, [1000, 10])

		y_ = tf.nn.softmax(dense_layer)

		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=self.y))

		# add an optimiser
		optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

		# define an accuracy assessment operation
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# setup the initialisation operator
		init_op = tf.global_variables_initializer()

		gpu_options = tf.GPUOptions(allow_growth=True)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
			allow_soft_placement=True)) as sess:
			# initialise the variables
			sess.run(init_op)
			total_batch = int(len(self.mnist.train.labels) / self.batch_size)
			for epoch in range(self.epochs):
				avg_cost = 0
				train_acc = 0
				test_acc = 0

				for i in range(total_batch):
					batch_x, batch_y = self.mnist.train.next_batch(batch_size=self.batch_size)
					_, c = sess.run([optimiser, cross_entropy],feed_dict={self.x: batch_x, self.y: batch_y})
					avg_cost += c / total_batch	
				
				test_acc = sess.run(accuracy, feed_dict={self.x:self.mnist.test.images, self.y: self.mnist.test.labels})
				print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "testing accuracy: {:.4f}".format(test_acc))
				

			print("\nTraining complete!")
			test_acc = sess.run(accuracy, feed_dict={self.x:self.mnist.test.images, self.y: self.mnist.test.labels})
			print('Testing accuracy: {:.4f}'.format(test_acc))



if __name__ == '__main__':
	myDenseNet().test_by_mnist()
