#This is the CNN model based on the paper: "Character-level Convolutional 
# Networks for Text Classification". Xiang Zhang, Junbo Zhao, Yann LeCun. 2015
# http://arxiv.org/abs/1509.01626

creppe_model <- function(vocab_size=69,num_output_classes=2){
  # Initial parameters
  kernel3 <- c(3,1)
  kernel7 <- c(7,1)
  stride <- c(1,1)
  num_filters <- 256
  fully_connected_size <- 1024
  pool_type <- "max"
  act_type <- "relu"
  drop <- 0.5
  #create model
  input_x <- mx.sym.Variable('data')  # placeholder for input
  input_y <- mx.sym.Variable('softmax_label')  # placeholder for output
  # 6 Convolutional layers
  # 1. alphabet x 1014
  conv1 <- mx.symbol.Convolution(data=input_x, kernel=c(7, vocab_size), 
                                 num_filter=num_filters)
  relu1 <- mx.symbol.Activation(data=conv1, act_type=act_type)
  pool1 <- mx.symbol.Pooling(data=relu1, pool_type=pool_type, kernel=kernel3, 
                             stride=stride)
  # 2. 336 x 256
  conv2 <- mx.symbol.Convolution(data=pool1, kernel=kernel7, 
                                 num_filter=num_filters)
  relu2 <- mx.symbol.Activation(data=conv2, act_type=act_type)
  pool2 <- mx.symbol.Pooling(data=relu2, pool_type=pool_type, kernel=kernel3, 
                             stride=stride)
  # 3. 110 x 256
  conv3 <- mx.symbol.Convolution(data=pool2, kernel=kernel3, 
                                 num_filter=num_filters)
  relu3 <- mx.symbol.Activation(data=conv3, act_type=act_type)
  # 4. 108 x 256
  conv4 <- mx.symbol.Convolution(data=relu3, kernel=kernel3, 
                                 num_filter=num_filters)
  relu4 <- mx.symbol.Activation(data=conv4, act_type=act_type)
  # 5. 106 x 256
  conv5 <- mx.symbol.Convolution(data=relu4, kernel=kernel3, 
                                 num_filter=num_filters)
  relu5 <- mx.symbol.Activation(data=conv5, act_type=act_type)
  # 6. 104 x 256
  conv6 <- mx.symbol.Convolution(data=relu5, kernel=kernel3, 
                                 num_filter=num_filters)
  relu6 <- mx.symbol.Activation(data=conv6, act_type=act_type)
  pool6 <- mx.symbol.Pooling(data=relu6, pool_type=pool_type, kernel=kernel3, 
                             stride=stride)
  # 34 x 256
  flatten <- mx.symbol.Flatten(data=pool6)
  # 3 Fully-connected layers
  # 7.  8704
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully_connected_size)
  act_fc1 <- mx.symbol.Activation(data=fc1, act_type=act_type)
  drop1 <- mx.sym.Dropout(act_fc1, p=drop)
  # 8. 1024
  fc2 <- mx.symbol.FullyConnected(data=drop1, num_hidden=fully_connected_size)
  act_fc2 <- mx.symbol.Activation(data=fc2, act_type=act_type)
  drop2 <- mx.sym.Dropout(act_fc2, p=drop)
  # 9. 1024
  fc3 <- mx.symbol.FullyConnected(data=drop2, num_hidden=num_output_classes)
  crepe <- mx.symbol.SoftmaxOutput(data=fc3, label=input_y, name="softmax")
  return(crepe)
}
