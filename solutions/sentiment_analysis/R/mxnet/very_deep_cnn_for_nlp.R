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
feature_len <- 1014
embedding_size <- 16
learning_rate <- 0.01 
momentum <- 0.9
batch_size <- 16 #in the paper was 128, but with that and GPUs it gets out of memory
######################################################################
# Fake input data
number_examples <- 200
input_fake <- as.integer(sample(c(0,1), replace=TRUE, size=vocab_size*feature_len*number_examples))
output_fake <- as.integer(sample(c(0,1), replace=TRUE, size=number_examples))
train.array <- input_fake
# Array dimension: it's width, height, channels, samples
dim(train.array) <- c(vocab_size, feature_len, 1, number_examples)
train.y <- output_fake
dim(train.array)
length(train.y)
format(object.size(train.array),units='auto')

######################################################################
#Real data
if(!exists("CustomCSVIter", mode="function")) source("CustomCSVIter.R")  
train.array <- CustomCSVIter(data.csv = "../../data/test_char_cnn_nohead.csv",
                          data.shape = c(vocab_size,feature_len + 1),
                          batch.size = batch_size,
                          shuffle = TRUE)

######################################################################
#Load model
if(!exists("network_model", mode="function")) source("network_model.R")  
network <- network_model()

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

