#Lenet CNN model
network_model <- function(){
  data <- mx.symbol.Variable('data')
  conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
  tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
  pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), 
                             stride=c(2,2))
  conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)# second conv
  tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
  pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), 
                             stride=c(2,2))
  flatten <- mx.symbol.Flatten(data=pool2)
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=100) # first fullc
  tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
  fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10) # second fullc
  network <- mx.symbol.SoftmaxOutput(data=fc2) # loss
  network
}