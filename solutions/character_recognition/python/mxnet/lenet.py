#http://yann.lecun.com/exdb/mnist/
import numpy as np
import pandas as pd
import mxnet as mx

path = '../../data/'

# gather data
train = pd.read_csv(path + 'mnist_train.csv', header=None)
train_y = train[[0]].values.ravel()
train_x = train.iloc[:,1:].values

# modify data
train_x = np.array(train_x, dtype='float32').reshape((-1, 1, 28, 28))
#print(train_x.shape)  # (60000, 1, 28, 28)
# normalise (between 0 and 1)
train_x[:] /= 255.0

# iterator to feed mini_batch at a time
# returns <mxnet.io.DataBatch object at 0x000001AA996B38D0> 
# type <class 'mxnet.io.DataBatch'>
train_iter = mx.io.NDArrayIter(train_x, train_y, batch_size=100, shuffle=True)

def create_lenet():
    # create symbolic representation
    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(
        data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(
        data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(
        data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(
        data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(
        data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2)) 

    flatten = mx.symbol.Flatten(
        data=pool2)
    fc1 = mx.symbol.FullyConnected(
        data=flatten, num_hidden=500) 
    tanh3 = mx.symbol.Activation(
        data=fc1, act_type="tanh")

    fc2 = mx.symbol.FullyConnected(
        data=tanh3, num_hidden=10) 

    lenet = mx.symbol.SoftmaxOutput(
        data=fc2, name="softmax")
    return lenet

# train the NN
ctx = mx.cpu()
cnn = create_lenet()

model = mx.model.FeedForward(
    ctx = ctx,
    symbol = cnn, 
    num_epoch = 10,
    learning_rate = 0.07,
    momentum = 0.9, 
    wd = 0.00001
    )

model.fit(X = train_iter)

# prediction of test set
test = pd.read_csv(path + 'mnist_test.csv', header=None)
test_y = test[[0]].values.ravel()
test_x = test.iloc[:,1:].values

test_x = np.array(test_x, dtype='float32').reshape((-1, 1, 28, 28))
test_x[:] /= 255.0

test_iter = mx.io.NDArrayIter(test_x, test_y, batch_size=100)

# most likely will be last element after sorting
pred = np.argsort(model.predict(X = test_iter))[:,-1]
# accuracy
sum(pred==test_y)/len(test_y) 