#http://yann.lecun.com/exdb/mnist/
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
test.x <- test[,-1]
test.y <- test[,1]
test.x <- t(test.x/255)
table(test.y)#number of examples
length(test.y)
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))


# Creation of LeNet
if(!exists("network_model", mode="function")) source("lenet_cnn_model.R")
lenet <- network_model()


#Train the NN
time_init <- Sys.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     wd=0.00001,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")


# Prediction of test set
preds <- predict(model, test.array)
pred.label = max.col(t(preds))-1
dim(preds)
length(pred.label)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(test.y, preds)))
