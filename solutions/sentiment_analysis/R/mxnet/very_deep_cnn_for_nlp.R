# Implementation of the paper Very Deep Convolutional Networks for Natual Language Processing, Conneau et al., 2016
library('mxnet')


# Initial parameters
kernel_size <- c(3,3)
stride_size <- c(2,2)
num_filters1 <- 64
num_filters2 <- num_filters1*2
num_filters3 <- num_filters2*2
num_filters4 <- num_filters3*2
fully_connected_size <- 2048
num_output_classes <- 2
vocab_size <- 69  
embedding_size <- 16
feature_len <- 16
learning_rate <- 0.01
momentum <- 0.9
minibatch_size <- 128
######################################################################
#input data
number_examples <- minibatch_size*2
input_fake <- sample(c(0,1), replace=TRUE, size=vocab_size*feature_len*number_examples)
output_fake <- sample(c(0,1), replace=TRUE, size=number_examples)
train.array <- input_fake
dim(train.array) <- c(vocab_size, feature_len, 1, number_examples)
train.y <- output_fake
######################################################################

# Model
data <- mx.symbol.Variable('data')
# Input = #vocab_size x 1014
#Embedding
#emb <- mx.symbol.Embedding(data=data, input.dim=vocab_size, output.dim=embedding_size)
# Initial convolution of size num_filters1
conv0 <- mx.symbol.Convolution(data=data, kernel=kernel_size, num_filter=num_filters1)
# First convolution block of size num_filters1 (x2)
conv1 <- mx.symbol.Convolution(data=conv0, kernel=kernel_size, num_filter=num_filters1)
norm1 <- mx.symbol.BatchNorm(data=conv1)
act1 <- mx.symbol.Activation(data=norm1, act_type="relu")
# Pooling/2 (=ROIPooling????)
pool1 <- mx.symbol.Pooling(data=act1, pool_type="max", kernel=kernel_size, stride=stride_size)
# Second convolution block of size num_filters2 (x2)
#conv2 <- mx.symbol.Convolution(data=pool1, kernel=kernel_size, num_filter=num_filters1)
#norm2 <- mx.symbol.BatchNorm(data=conv2)
#act2 <- mx.symbol.Activation(data=norm2, act_type="relu")
# Pooling/2
#pool2 <- mx.symbol.Pooling(data=act2, pool_type="max", kernel=kernel_size, stride=stride_size)
#....
#....
flatten <- mx.symbol.Flatten(data=pool1)
# first fullc
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully_connected_size) 
act_fc1 <- mx.symbol.Activation(data=fc1, act_type="relu")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=act_fc1, num_hidden=fully_connected_size)
act_fc1 <- mx.symbol.Activation(data=fc1, act_type="relu")
# third fullc
fc3 <- mx.symbol.FullyConnected(data=act_fc1, num_hidden=num_output_classes)
model <- mx.symbol.SoftmaxOutput(data=fc3) # loss

#Train the NN
devices <- mx.cpu()
time_init <- Sys.time()
model <- mx.model.FeedForward.create(model, X=train.array, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=minibatch_size,
                                     learning.rate=learning_rate, momentum=momentum,  
                                     eval.metric=mx.metric.accuracy, wd=0.00001,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")


