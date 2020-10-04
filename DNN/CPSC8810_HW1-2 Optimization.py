
# coding: utf-8

# # CPSC8810_HW1-2 Optimization

# ## Import

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from sklearn.decomposition import PCA
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}b


# In[2]:


tf.__version__


# ## Simulated Function
# #### f(x) = f(x) = sgn(sin(5πx))

# In[3]:


X = np.arange(0.0001,1,0.0001)
X_train = X.reshape(-1,1).astype(np.float32)
Y = np.sign(np.sin(5*np.pi*X))
Y_train = Y.reshape(-1,1).astype(np.float32)
plt.plot(X,Y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = f(x) = f(x) = sgn(sin(5πx))');


# ## Placeholder variables

# In[4]:


tf.reset_default_graph()
tf.random.set_random_seed(3)
np.random.seed(1)
input_x = tf.placeholder(tf.float32,shape=[None,1])
output_y = tf.placeholder(tf.float32,shape=[None,1])


# ## Training Model

# In[5]:


# Model 1 neural network layers
h1 = tf.layers.dense(inputs=input_x, units=5, activation=tf.nn.relu, name='h1')   # Input layer
h2 = tf.layers.dense(inputs=h1, units=10, activation=tf.nn.relu, name='h2')        # hidden layer
h3 = tf.layers.dense(inputs=h2, units=15, activation=tf.nn.relu, name='h3')        # hidden layer
h4 = tf.layers.dense(inputs=h3, units=10, activation=tf.nn.relu, name='h4')        # hidden layer
h5 = tf.layers.dense(inputs=h3, units=5, activation=tf.nn.relu, name='h5')        # hidden layer
output = tf.layers.dense(inputs=h4, units=1, name='output')    


# ## Loss Function

# In[6]:


loss = tf.losses.mean_squared_error(output_y, output)   # compute cost
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)


# ## Get weight

# In[7]:


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

w_h1 = tf.reshape(get_weights_variable('h1'),[1,-1])
w_h2 = tf.reshape(get_weights_variable('h2'),[1,-1])
w_h3 = tf.reshape(get_weights_variable('h3'),[1,-1])
w_h4 = tf.reshape(get_weights_variable('h4'),[1,-1])
w_h5 = tf.reshape(get_weights_variable('h5'),[1,-1])
w_out = tf.reshape(get_weights_variable('output'),[1,-1])
w_model = tf.concat([w_h1,w_h2,w_h3,w_h4,w_h5,w_out],axis=1)


# In[8]:


#grad = opt.compute_gradients(loss, weights_fc_out)[0]
grads = tf.gradients(loss, get_weights_variable('output'))[0]
grads_norm = tf.norm(grads)
print(grads)
print(grads_norm)

hessian = tf.reduce_sum(tf.hessians(loss, get_weights_variable('output'))[0], axis = 2)
print(hessian)

eigenvalue = tf.linalg.eigvalsh(hessian)
minimal_ratio = tf.divide(tf.count_nonzero(tf.greater(eigenvalue, 0.)),eigenvalue.shape[0])
print(minimal_ratio)


# ## Training using normal loss function

# In[9]:


sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph
EPOCH = 1000

loss_list = []
grads_norm_list = []
min_ratio_list = []
for i in range(EPOCH):
    # train and net output
    _, l, pred, grad_norm,min_ratio = sess.run([train_op, loss, output,grads_norm,minimal_ratio], feed_dict={input_x: X_train, output_y: Y_train})
    loss_list.append(l)
    grads_norm_list.append(grad_norm)
    min_ratio_list.append(min_ratio)
    if i%50 == 0:
        print("Epoch: ",i,"Loss: ",l)


# ## Observe Gradient Norm During Training

# In[16]:


plt.plot(grads_norm_list)
plt.title('Plot Grad Norm')
plt.ylabel('grad norm')
plt.xlabel('Epoch')
plt.show()


# ## Plot Loss

# In[17]:


plt.plot(loss_list)
plt.title('Plot Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# ## Plot Minimal Ratio

# In[18]:


plt.pause(0.1)
plt.scatter(min_ratio_list,loss_list)
plt.xlabel('Minimal Ratio')
plt.ylabel('Loss');
plt.title('Minimal Ratio vs Loss')


# ## Training using normal loss function

# In[11]:


grad_norm_train_op = optimizer.minimize(grads_norm)


# In[12]:


EPOCH = 100
grads_loss_list = []
min_ratio_list2 = []
for i in range(EPOCH):
    # train and net output
    _, l, min_ratio = sess.run([grad_norm_train_op, loss, minimal_ratio], feed_dict={input_x: X_train, output_y: Y_train})
    min_ratio_list2.append(min_ratio)
    grads_loss_list.append(l)
    if i%10 == 0:
        print("Epoch: ",i,"Loss: ",l,"Minimal Ratio: ",min_ratio)


# In[13]:


plt.scatter(min_ratio_list2,grads_loss_list)
plt.xlabel('Minimal Ratio')
plt.ylabel('Loss')

