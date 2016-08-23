# Example of lenet network with using an advance training method
require(mxnet)

#Load the data as an image iterator
root_path <- c("../../data/")
data_shape <- c(28, 28, 1)
batch.size <- 256
dtrain <- mx.io.MNISTIter(
  image=file.path(root_path,"train-images-idx3-ubyte"),
  label=file.path(root_path,"train-labels-idx1-ubyte"),
  data.shape=data_shape,
  batch.size=batch.size,
  shuffle=TRUE,
  flat=FALSE,
  silent=0,
  seed=10)

dtest <- mx.io.MNISTIter(
  image=file.path(root_path,"t10k-images-idx3-ubyte"),
  label=file.path(root_path,"t10k-labels-idx1-ubyte"),
  data.shape=data_shape,
  batch.size=batch.size,
  shuffle=FALSE,
  flat=FALSE,
  silent=0)

# Network configuration
if(!exists("network_model", mode="function")) source("lenet_cnn_model.R")
lenet <- network_model()

# Devices
devices = lapply(1:2, function(i) {
  mx.cpu(i)
})
#devices <- mx.gpu()

# create the model and train
time_init <- Sys.time()
model <- mx.model.FeedForward.create(symbol=lenet, X=dtrain, eval.data=dtest,
                                     ctx=devices, num.round=1,
                                     learning.rate=0.1, momentum=0.9,
                                     eval.metric = mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.save.checkpoint("chkpt"),
                                     batch.end.callback=mx.callback.log.train.metric(50))
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")

# do prediction
pred <- predict(model, dtest)
label <- mx.io.extract(dtest, "label")
dataX <- mx.io.extract(dtest, "data")
# Predict with R's array
pred2 <- predict(model, X=dataX)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(label, pred)))
print(paste0("Finish prediction... accuracy2=", accuracy(label, pred2)))


# load the model
model <- mx.model.load("chkpt", 1)

#continue training with some new arguments
time_init <- Sys.time()
model <- mx.model.FeedForward.create(model$symbol, X=dtrain, eval.data=dtest,
                                     ctx=devices, num.round=9,
                                     learning.rate=0.1, momentum=0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.save.checkpoint("reload_chkpt"),
                                     batch.end.callback=mx.callback.log.train.metric(50),
                                     arg.params=model$arg.params, aux.params=model$aux.params)
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")

# do prediction
pred <- predict(model, dtest)
label <- mx.io.extract(dtest, "label")
dataX <- mx.io.extract(dtest, "data")
# Predict with R's array
pred2 <- predict(model, X=dataX)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(label, pred)))
print(paste0("Finish prediction... accuracy2=", accuracy(label, pred2)))



