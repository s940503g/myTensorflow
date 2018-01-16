#import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.0001
epochs = 10
batch_size = 50
dropout_rate = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, 784])
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

def create_conv_layer(input_data, num_input_channels, num_filters, filter_shape, dropout_rate, name):

	conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
	weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
	out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

	# add bias
	bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
	out_layer = tf.nn.bias_add(out_layer, bias)

	# apply a ReLU non-linear activation
	out_layer = tf.nn.relu(out_layer)
	return out_layer
'''
	# apply a ReLU dropout
	if dropout_rate:
		out_layer = tf.nn.dropout(out_layer, 1-dropout_rate)
'''


def create_max_pooling(input_layer, pool_shape, name):
	ksize = [1, pool_shape[0], pool_shape[1], 1]
	strides = [1, 2, 2, 1]
	out_layer = tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding='SAME')

	return out_layer

def create_dense_layer(input_layer, hidden_layer, name):
	input_layer_shape = input_layer.get_shape().as_list()
	input_nodes = input_layer_shape[1] * input_layer_shape[2] * input_layer_shape[3]
	flattened = tf.reshape(input_layer, [-1, input_nodes])

	wd1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer[0]], stddev=0.03), name=name+'_wd')
	bd1 = tf.Variable(tf.truncated_normal([hidden_layer[0]], stddev=0.01), name=name+'_bd')

	dense_layer1 = tf.matmul(flattened, wd1) + bd1
	dense_layer1 = tf.nn.relu(dense_layer1)

	wd2 = tf.Variable(tf.truncated_normal(hidden_layer, stddev=0.03), name='wd2')
	bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
	dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2

	return dense_layer2

# create some convolutional layers

layer1 = create_conv_layer(x_shaped, 1, 50, [5, 5], dropout_rate=0, name='layer1_conv')
layer1 = create_max_pooling(layer1, [2,2], name='layer1_max_pooling')
print("layer1: ", layer1.get_shape());

layer2 = create_conv_layer(layer1, 50, 50, [5, 5], dropout_rate=0, name='layer2_conv')
layer2 = create_max_pooling(layer2, [2,2], name='layer2_max_pooling')
print("layer2: ", layer2.get_shape());

layer3 = create_conv_layer(layer2, 50, 50, [5, 5], dropout_rate=0.3, name='layer3_conv')
layer3 = create_max_pooling(layer3, [2,2], name='layer3_max_pooling')
print("layer3: ", layer3.get_shape());

layer4 = create_conv_layer(layer3, 50, 50, [5, 5], dropout_rate=0.3, name='layer4_conv')
layer4 = create_max_pooling(layer4, [2,2], name='layer4_max_pooling')
print("layer4: ", layer4.get_shape());

layer5 = create_conv_layer(layer4, 50, 50, [5, 5], dropout_rate=0.3, name='layer5_conv')
layer5 = create_max_pooling(layer5, [2,2], name='layer5_max_pooling')
print("layer5: ", layer5.get_shape());

dense_layer = create_dense_layer(layer5, [1000, 10], name="dense_layer")

y_ = tf.nn.softmax(dense_layer)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=y))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()
print("start.")

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)) as sess:
	# initialise the variables
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		total_batch = 200
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimiser, cross_entropy],
							feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		train_acc = sess.run(accuracy,feed_dict={x: batch_x, y: batch_y})
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " training accuracy: {:.3f}".format(train_acc))
		#train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})

	print("\nTraining complete!")
	test_x, test_y = mnist.test.next_batch(batch_size=batch_size)
	print(sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
'''
		test_acc = sess.run(accuracy,
					   feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

	print("\nTraining complete!")
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
'''
