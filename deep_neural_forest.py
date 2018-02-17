import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

# Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128
display_step = 100

depth = 3                 # Depth of a tree
n_leaf = 2 ** (depth + 1)  # Number of leaf node
n_label = 10                # Number of classes
n_tree = 5                 # Number of trees (ensemble)
n_batch = 128               # Number of data points per mini-batch

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
num_epochs = 100

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
	return tf.Variable(tf.random_uniform(shape, minval, maxval))


def model(X, w, w2, w3, w4_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):
	"""
	Create a forest and return the neural decision forest outputs:
		decision_p_e: decision node routing probability for all ensemble
			If we number all nodes in the tree sequentially from top to bottom,
			left to right, decision_p contains
			[d(0), d(1), d(2), ..., d(2^n - 2)] where d(1) is the probability
			of going left at the root node, d(2) is that of the left child of
			the root node.
			decision_p_e is the concatenation of all tree decision_p's
		leaf_p_e: terminal node probability distributions for all ensemble. The
			indexing is the same as that of decision_p_e.
	"""
	assert(len(w4_e) == len(w_d_e))
	assert(len(w4_e) == len(w_l_e))

	X = tf.reshape(X, shape=[-1, 28, 28, 1])

	l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
	l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')
	l1 = tf.nn.dropout(l1, p_keep_conv)

	l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
	l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')
	l2 = tf.nn.dropout(l2, p_keep_conv)

	l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
	l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

	l3 = tf.reshape(l3, [-1, w4_e[0].get_shape().as_list()[0]])
	l3 = tf.nn.dropout(l3, p_keep_conv)

	decision_p_e = []
	leaf_p_e = []
	for w4, w_d, w_l in zip(w4_e, w_d_e, w_l_e):
		l4 = tf.nn.relu(tf.matmul(l3, w4))
		l4 = tf.nn.dropout(l4, p_keep_hidden)

		decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d))
		leaf_p = tf.nn.softmax(w_l)

		decision_p_e.append(decision_p)
		leaf_p_e.append(leaf_p)

	return decision_p_e, leaf_p_e

#################################################
# Load dataset
##################################################
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

sess = tf.Session()

# Create a dataset tensor from the images and the labels
dataset = tf.contrib.data.Dataset.from_tensor_slices(
	(mnist.train.images, mnist.train.labels)
)
dataset = dataset.repeat(num_epochs)
dataset = dataset.shuffle(1000)
# Create batches of data
dataset = dataset.batch(batch_size)
# Create an iterator, to go over the dataset
iterator = dataset.make_initializable_iterator()
_data = tf.placeholder(tf.float32, [None, n_input])
_labels = tf.placeholder(tf.float32, [None, n_classes])
# and avoid the 2Gb restriction length of a tensor.
# Input X, output Y
sess.run(
	iterator.initializer,
	feed_dict={_data: mnist.train.images, _labels: mnist.train.labels}
)
X, Y = iterator.get_next()
Y = tf.cast(Y,tf.float32)


##################################################
# Initialize network weights
##################################################
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])

w4_ensemble = []
w_d_ensemble = []
w_l_ensemble = []
for i in range(n_tree):
	w4_ensemble.append(init_weights([128 * 4 * 4, 625]))
	w_d_ensemble.append(init_prob_weights([625, n_leaf], -1, 1))
	w_l_ensemble.append(init_prob_weights([n_leaf, n_label], -2, 2))

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

##################################################
# Define a fully differentiable deep-ndf
##################################################
# With the probability decision_p, route a sample to the right branch
decision_p_e, leaf_p_e = model(
	X, w, w2, w3, w4_ensemble, w_d_ensemble,
	w_l_ensemble, p_keep_conv, p_keep_hidden
)

flat_decision_p_e = []

# iterate over each tree
for decision_p in decision_p_e:
	# Compute the complement of d, which is 1 - d
	# where d is the sigmoid of fully connected output
	decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)

	# Concatenate both d, 1-d
	decision_p_pack = tf.stack([decision_p, decision_p_comp])

	# Flatten/vectorize the decision probabilities for efficient indexing
	flat_decision_p = tf.reshape(decision_p_pack, [-1])
	flat_decision_p_e.append(flat_decision_p)

# 0 index of each data instance in a mini-batch
batch_0_indices = tf.tile(
	tf.expand_dims(tf.range(0, n_batch * n_leaf, n_leaf), 1), [1, n_leaf]
)

###############################################################################
# The routing probability computation
#
# We will create a routing probability matrix \mu. First, we will initialize
# \mu using the root node d, 1-d. To efficiently implement this routing, we
# will create a giant vector (matrix) that contains all d and 1-d from all
# decision nodes. The matrix version of that is decision_p_pack and vectorized
# version is flat_decision_p.
#
# The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
# from trees.
#
# For depth = 2 tree, the routing probability for each leaf node can be easily
# compute by multiplying the following vectors elementwise.
# \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
# \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
# \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
#
# Tree indexing
#      0
#    1   2
#   3 4 5 6
##############################################################################
in_repeat = int(n_leaf / 2)
out_repeat = int(n_batch)

# Let n_batch * n_leaf be N_D. flat_decision_p[N_D] will return 1-d of the
# first root node in the first tree.
batch_complement_indices = np.array(
	[[0] * in_repeat, [n_batch * n_leaf] * in_repeat] * out_repeat
).reshape(n_batch, n_leaf)

# First define the routing probabilities d for root nodes
mu_e = []

# iterate over each tree
for i, flat_decision_p in enumerate(flat_decision_p_e):
	mu = tf.gather(flat_decision_p,
				   tf.add(batch_0_indices, batch_complement_indices))
	mu_e.append(mu)

# from the second layer to the last layer, we make the decision nodes
for d in range(1, depth + 1):
	indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
	tile_indices = tf.reshape(
		tf.tile(tf.expand_dims(indices, 1), [1, 2 ** (depth - d + 1)]), [1, -1]
	)
	batch_indices = tf.add(
		batch_0_indices,
		tf.tile(tile_indices, [n_batch, 1])
	)

	in_repeat = int(in_repeat / 2)
	out_repeat = int(out_repeat * 2)

	# Again define the indices that picks d and 1-d for the node
	batch_complement_indices = np.array(
		[[0] * in_repeat, [n_batch * n_leaf] * in_repeat] * out_repeat
	).reshape(n_batch, n_leaf)

	mu_e_update = []
	for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
		mu = tf.multiply(
			mu,
			tf.gather(
				flat_decision_p,
				tf.add(batch_indices, batch_complement_indices)
			)
		)
		mu_e_update.append(mu)

	mu_e = mu_e_update

##################################################
# Define p(y|x)
##################################################
py_x_e = []
for mu, leaf_p in zip(mu_e, leaf_p_e):
	# average all the leaf p
	py_x_tree = tf.reduce_mean(
		tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, n_label]),
			   tf.tile(tf.expand_dims(leaf_p, 0), [n_batch, 1, 1])), 1)
	py_x_e.append(py_x_tree)

py_x_e = tf.stack(py_x_e)
py_x = tf.reduce_mean(py_x_e, 0)

##################################################
# Define cost and optimization method
##################################################

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(-tf.multiply(tf.log(py_x), Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)

# Training cycle
for step in range(1, num_steps + 1):

	try:
		# Run optimization
		sess.run(train_op,
			feed_dict={p_keep_conv: 0.8, p_keep_hidden: 0.5}
		)
	except tf.errors.OutOfRangeError:
		# Reload the iterator when it reaches the end of the dataset
		sess.run(iterator.initializer,
				 feed_dict={_data: mnist.train.images,
							_labels: mnist.train.labels})
		sess.run(train_op)

	if step % display_step == 0 or step == 1:
		# Calculate batch loss and accuracy
		# (note that this consume a new batch of data)
		loss, acc = sess.run([loss_op, accuracy],
			feed_dict={
				p_keep_conv: 0.8,
				p_keep_hidden: 0.5}
		)
		print("Step " + str(step) + ", Minibatch Loss= " + \
			  "{:.4f}".format(loss) + ", Training Accuracy= " + \
			  "{:.3f}".format(acc))

print("Optimization Finished!")