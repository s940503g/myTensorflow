import numpy
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

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
	# setup the filter input shape for tf.nn.conv_2d
	conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
	# initialise weights and bias for the filter
	weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
	bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

	# setup the convolutional layer operation
	out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

	# add the bias
	out_layer += bias

	# apply a ReLU non-linear activation
	out_layer = tf.nn.relu(out_layer)

	# now perform max pooling
	ksize = [1, pool_shape[0], pool_shape[1], 1]
	strides = [1, 2, 2, 1]
	out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

	return out_layer
	
def create_new_conv_layer_without_max_pool(input_data, num_input_channels, num_filters, filter_shape, name):
	# setup the filter input shape for tf.nn.conv_2d
	conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
										num_filters]
	# initialise weights and bias for the filter
	weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
	bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

	# setup the convolutional layer operation
	out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

	# add the bias
	out_layer += bias

	# apply a ReLU non-linear activation
	out_layer = tf.nn.relu(out_layer)

	return out_layer


numpy.random.seed(123)
# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [3, 3], [2,2], name='layer1')
print("layer1: ", layer1.get_shape());

layer2 = create_new_conv_layer(layer1, 32, 64, [3, 3], [2,2], name='layer2')
print("layer2: ", layer2.get_shape());

layer3 = create_new_conv_layer(layer2, 64, 128, [3, 3], [2,2], name='layer3')
print("layer3: ", layer3.get_shape());

layer4 = create_new_conv_layer(layer3, 128, 256, [3, 3], [2,2], name='layer4')
print("layer4: ", layer4.get_shape());

layer5 = create_new_conv_layer(layer4, 256, 512, [3, 3], [2,2], name='layer5')
print("layer5: ", layer5.get_shape());

flattened = tf.reshape(layer5, [-1, 1 * 1 * 512])
print("flattened: ", flattened.get_shape())

# concat

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([1 * 1 * 512, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')

dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2

# dropout
#dense_layer2 = tf.nn.dropout(dense_layer2, keep_prob=0.9)

y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	# initialise the variables
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimiser, cross_entropy], 
							feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		test_acc = sess.run(accuracy, 
					   feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

	print("\nTraining complete!")
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))