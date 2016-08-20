
require(mxnet)

#Initialize variables
devices <- mx.cpu()
mx.set.seed(0)


#Gather the data
root_path <- c("../../data/")
train_file <- file.path(root_path,"mnist_train.csv")
test_file <- file.path(root_path,"mnist_test.csv")
train <- read.csv(train_file, header=TRUE)
train <- data.matrix(train)
test <- read.csv(test_file, header=TRUE)
test <- data.matrix(test)
dim(train)
dim(test)


#Modify data
train.x <- train[,-1]
train.y <- train[,1]
train.x <- t(train.x/255)
table(train.y)#number of examples
length(train.y)


#Create the NN
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")


#Train the NN
time_init <- Sys.time()
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")


# Prediction of test set
test.x <- test[,-1]
test.y <- test[,1]
test.x <- t(test.x/255)
table(test.y)#number of examples
length(test.y)
preds <- predict(model, test.x)
pred.label = max.col(t(preds))-1
dim(preds)
length(pred.label)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(test.y, preds)))

