# Convolution block
convolution_block <- function(data, kernel, num_filter, act_type){
  conv <- mx.symbol.Convolution(data=data, kernel=kernel, num_filter=num_filter)
  norm <- mx.symbol.BatchNorm(data=conv)
  act <- mx.symbol.Activation(data=norm, act_type=act_type)
}

# Complete Network
network_model <- function(){
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
  pool_type <- "max"
  act_type <- "relu"
  # Model
  data <- mx.symbol.Variable('data')
  # Input = #vocab_size x 1014
  conv0 <- mx.symbol.Convolution(data=data, num_filter=num_filters1, kernel=kernel)
  # First convolution block of size num_filters1 (x2)
  conv11 <- convolution_block(data = data, kernel=kernel, num_filter=num_filters1, act_type=act_type)
  conv12 <- convolution_block(data = conv11, kernel=kernel, num_filter=num_filters1, act_type=act_type)
  # Pooling/2 
  pool1 <- mx.symbol.Pooling(data=conv12, pool_type=pool_type, kernel=kernel, stride=stride)
  # Second convolution block of size num_filters2 (x2)
  conv21 <- convolution_block(data = pool1, kernel=kernel, num_filter=num_filters2, act_type=act_type)
  conv22 <- convolution_block(data = conv21, kernel=kernel, num_filter=num_filters2, act_type=act_type)
  # Pooling/2
  pool2 <- mx.symbol.Pooling(data=conv22, pool_type=pool_type, kernel=kernel, stride=stride)
  # Thrid convolution block of size num_filters3 (x2)
  conv31 <- convolution_block(data = pool2, kernel=kernel, num_filter=num_filters3, act_type=act_type)
  conv32 <- convolution_block(data = conv11, kernel=kernel, num_filter=num_filters3, act_type=act_type)
  # Pooling/2
  pool3 <- mx.symbol.Pooling(data=conv22, pool_type=pool_type, kernel=kernel, stride=stride)
  # Fourth convolution block of size num_filters4 (x2)
  conv41 <- convolution_block(data = pool3, kernel=kernel, num_filter=num_filters4, act_type=act_type)
  conv42 <- convolution_block(data = conv41, kernel=kernel, num_filter=num_filters4, act_type=act_type)
  # Pooling k=8
  pool4 <- mx.symbol.Pooling(data=conv42, pool_type=pool_type, kernel=kernel_out, stride=stride)
  # First fully connected
  flatten <- mx.symbol.Flatten(data=pool4)
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully_connected_size) 
  act_fc1 <- mx.symbol.Activation(data=fc1, act_type=act_type)
  # Second fully connected
  fc2 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully_connected_size)
  act_fc1 <- mx.symbol.Activation(data=fc2, act_type=act_type)
  # third fullc
  fc3 <- mx.symbol.FullyConnected(data=flatten, num_hidden=num_output_classes)
  network <- mx.symbol.SoftmaxOutput(data=fc3) # loss
  
}