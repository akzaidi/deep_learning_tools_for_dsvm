# Implementation of the paper Very Deep Convolutional Networks for Natual Language Processing, Conneau et al., 2016
library('mxnet')

# Initial parameters
kernel <- c(3,3)
kernel_out <- c(8,8)
stride <- c(2,2)
num_filters1 <- 64
num_filters2 <- num_filters1*2
num_filters3 <- num_filters2*2
num_filters4 <- num_filters3*2
fully_connected_size <- 2048
num_output_classes <- 2
vocab_size <- 69  
embedding_size <- 16
feature_len <- 1014
learning_rate <- 0.01 
momentum <- 0.9
batch_size <- 100 #in the paper was 128, but with that and GPUs it gets out of memory
######################################################################
#input data
number_examples <- batch_size*2
input_fake <- as.integer(sample(c(0,1), replace=TRUE, size=vocab_size*feature_len*number_examples))
output_fake <- as.integer(sample(c(0,1), replace=TRUE, size=number_examples))
train.array <- input_fake
# array dimension: it's width, height, channels, samples
dim(train.array) <- c(vocab_size, feature_len, 1, number_examples)
train.y <- output_fake
dim(train.array)
length(train.y)
format(object.size(train.array),units='auto')

# CSVIter is uesed here, since the data can't fit into memory
data_train <- mx.io.CSVIter(
  data.csv = "./train-64x64-data.csv", data.shape = c(64, 64, 30),
  label.csv = "./train-stytole.csv", label.shape = 600,
  batch.size = batch_size
)
######################################################################
# Convolution block
convolution_block <- function(data, kernel, num_filter, act_type = 'relu'){
  conv <- mx.symbol.Convolution(data=data, kernel=kernel, num_filter=num_filter)
  norm <- mx.symbol.BatchNorm(data=conv)
  act <- mx.symbol.Activation(data=norm, act_type="relu")
}
  
# Model
#API info: https://turi.com/products/create/docs/graphlab.mxnet.html
data <- mx.symbol.Variable('data')
# Input = #vocab_size x 1014
conv0 <- mx.symbol.Convolution(data=data, num_filter=num_filters1, kernel=kernel)
# First convolution block of size num_filters1 (x2)
conv11 <- convolution_block(data = data, kernel=kernel, num_filter=num_filters1)
conv12 <- convolution_block(data = conv11, kernel=kernel, num_filter=num_filters1)
# Pooling/2 (=ROIPooling????)
pool1 <- mx.symbol.Pooling(data=conv12, pool_type="max", kernel=kernel, stride=stride)
# Second convolution block of size num_filters2 (x2)
conv21 <- convolution_block(data = pool1, kernel=kernel, num_filter=num_filters2)
conv22 <- convolution_block(data = conv21, kernel=kernel, num_filter=num_filters2)
# Pooling/2
pool2 <- mx.symbol.Pooling(data=conv22, pool_type="max", kernel=kernel, stride=stride)
# Thrid convolution block of size num_filters3 (x2)
conv31 <- convolution_block(data = pool2, kernel=kernel, num_filter=num_filters3)
conv32 <- convolution_block(data = conv11, kernel=kernel, num_filter=num_filters3)
# Pooling/2
pool3 <- mx.symbol.Pooling(data=conv22, pool_type="max", kernel=kernel, stride=stride)
# Fourth convolution block of size num_filters4 (x2)
conv41 <- convolution_block(data = pool3, kernel=kernel, num_filter=num_filters4)
conv42 <- convolution_block(data = conv41, kernel=kernel, num_filter=num_filters4)
# Pooling k=8
pool4 <- mx.symbol.Pooling(data=conv42, pool_type="max", kernel=kernel_out, stride=stride)
# First fully connected
flatten <- mx.symbol.Flatten(data=pool4)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully_connected_size) 
act_fc1 <- mx.symbol.Activation(data=fc1, act_type="relu")
# Second fully connected
fc2 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully_connected_size)
act_fc1 <- mx.symbol.Activation(data=fc2, act_type="relu")
# third fullc
fc3 <- mx.symbol.FullyConnected(data=flatten, num_hidden=num_output_classes)
network <- mx.symbol.SoftmaxOutput(data=fc3) # loss

#Train the NN
devices <- mx.cpu()
time_init <- Sys.time()
model <- mx.model.FeedForward.create(network, X=train.array, y=train.y,
                                     ctx=devices, num.round=1, array.batch.size=batch_size,
                                     learning.rate=learning_rate, momentum=momentum,  
                                     eval.metric=mx.metric.accuracy, wd=0.00001,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")

#gc()

