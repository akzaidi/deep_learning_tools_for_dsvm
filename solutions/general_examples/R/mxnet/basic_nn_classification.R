require(mlbench)
require(mxnet)
#Initialize variables
devices <- mx.cpu() #mx.gpu()
mx.set.seed(0)
mx.ctx.internal.default.value = list(device="cpu",device_id=0,device_typeid=1)
class(mx.ctx.internal.default.value) = "MXContext"

data(Sonar, package="mlbench")

Sonar[,61] = as.numeric(Sonar[,61])-1
train.ind = c(1:50, 100:150)
train.x = data.matrix(Sonar[train.ind, 1:60])
train.y = Sonar[train.ind, 61]
test.x = data.matrix(Sonar[-train.ind, 1:60])
test.y = Sonar[-train.ind, 61]

time_init <- Sys.time()
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                eval.metric=mx.metric.accuracy)
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")

graph.viz(model$symbol$as.json())
preds = predict(model, test.x)
pred.label = max.col(t(preds))-1
table(pred.label, test.y)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(test.y, preds)))
