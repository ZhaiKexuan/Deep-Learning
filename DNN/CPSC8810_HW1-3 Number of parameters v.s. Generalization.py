
# coding: utf-8

# # CPSC8810_HW1-3 Number of parameters v.s. Generalization

# In[1]:


import tensorflow as tf
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA

tf.__version__
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


data = input_data.read_data_sets('data/MNIST/', one_hot=True);

train_num = data.train.num_examples
valid_num = data.validation.num_examples
test_num = data.test.num_examples
img_flatten = 784
img_size = 28
num_classes = 10
print("Training Dataset Size:",train_num)
print("Validation Dataset Size:",valid_num)
print("Testing Dataset Size:",test_num)


# In[3]:


def parameter_count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        #print(variable)
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        #print("parameter num:",variable_parameters)
        total_parameters += variable_parameters
    return total_parameters


# ## Model 1 Architecure

# In[4]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=2,kernel_size=1,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=2,kernel_size=1,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m1 = parameter_count()
print('m1 = ', m1)


# ## Training Model 1

# In[5]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 2 Architecure

# In[6]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=2,kernel_size=1,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=1)

conv2 = tf.layers.conv2d(inputs=pool1,filters=3,kernel_size=2,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m2 = parameter_count()
print('m2 = ', m2)


# ## Training Model 2

# In[7]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64

for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 3 Architecure

# In[8]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=2,kernel_size=2,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=4,kernel_size=2,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m3 = parameter_count()
print('m3 = ', m3)


# ## Training Model 3

# In[9]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64

for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 4 Architecure

# In[10]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=4,kernel_size=3,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=4,kernel_size=3,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m4 = parameter_count()
print('m4 = ', m4)


# ## Training Model 4

# In[11]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64


for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 5 Architecure

# In[12]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=4,kernel_size=4,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=8,kernel_size=4,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m5 = parameter_count()
print('m5 = ', m5)


# ## Training Model 5

# In[13]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64

for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 6 Architecure

# In[14]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=3,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=8,kernel_size=3,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m6 = parameter_count()
print('m6 = ', m6)


# ## Training Model 6

# In[15]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64



for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 7 Architecure

# In[16]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=12,kernel_size=5,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m7 = parameter_count()
print('m7 = ', m7)


# ## Training Model 7

# In[17]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64



for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 8 Architecure

# In[18]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m8 = parameter_count()
print('m8 = ', m8)


# ## Train Model 8

# In[19]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64



for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 9 Architecure

# In[20]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=10,kernel_size=5,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=10,kernel_size=5,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m9 = parameter_count()
print('m9 = ', m9)


# ## Training Model 9

# In[21]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64



for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Model 10 Architecure

# In[22]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape = [None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=16,kernel_size=7,padding="SAME",activation=tf.nn.leaky_relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=7,padding="same",activation=tf.nn.leaky_relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.leaky_relu)
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits)
pred_op = tf.argmax(softmax,dimension=1)
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = opt.minimize(loss)

m10 = parameter_count()
print('m10 = ', m10)


# ## Training Model 10

# In[23]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph

EPOCH = 1
BATCH_SIZE = 64

for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_true_batch})


    train_loss, train_acc = sess.run([loss,acc_op], feed_dict={x: x_batch,y: y_true_batch})
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss, test_acc = sess.run([loss,acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # Message for printing.
    msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    # Print it.
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))


# ## Plot

# In[24]:


# Plot loss
plt.scatter([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10],train_loss_list)
plt.scatter([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10],test_loss_list)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Number of Parameters');

plt.pause(0.1)

# Plot accuracy
plt.scatter([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10],train_acc_list)
plt.scatter([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10],test_acc_list)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Parameters');

