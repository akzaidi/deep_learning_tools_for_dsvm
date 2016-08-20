require(mxnet)
require(argparse)


get_iterator <- function(data.shape) {
    data_dir = args$data_dir
    data.shape <- data.shape
    train = mx.io.ImageRecordIter(
      path.imgrec     = paste0(data_dir, "train.rec"),
      path.imglist    = paste0(data_dir, "train.lst"),
      batch.size      = args$batch_size,
      data.shape      = data.shape,
      shuffle         = TRUE
    )
    
    val = mx.io.ImageRecordIter(
      path.imgrec     = paste0(data_dir, "test.rec"),
      path.imglist    = paste0(data_dir, "test.lst"),
      batch.size      = args$batch_size,
      data.shape      = data.shape
    )
    ret = list(train=train, value=val)
}

parse_args <- function() {
  parser <- ArgumentParser(description='train an image classifer on CIFAR')
  parser$add_argument('--network', type='character', default='resnet-28-small',
                      choices = c('mlp', 'lenet'),
                      help = 'the cnn to use')
  parser$add_argument('--data-dir', type='character', default='../../data/cifar10/',
                      help='the input data directory')
  parser$add_argument('--gpus', type='character',
                      help='the gpus will be used, e.g "0,1,2,3"')
  parser$add_argument('--batch-size', type='integer', default=128,
                      help='the batch size')
  parser$add_argument('--lr', type='double', default=.1,
                      help='the initial learning rate')
  parser$add_argument('--model-prefix', type='character',
                      help='the prefix of the model to load/save')
  parser$add_argument('--num-round', type='integer', default=10,
                      help='the number of iterations over training data to train the model')
  parser$add_argument('--kv-store', type='character', default='local',
                      help='the kvstore type')
  parser$parse_args()
}
args = parse_args()

# train
if (args$network == 'resnet') {
  source("symbol_resnet.R")
  net <- get_symbol()
} else {
  source("symbol_resnet-28-small.R")
  net <- get_symbol()
}

# save model
if (is.null(args$model_prefix)) {
  checkpoint <- NULL
} else {
  checkpoint <- mx.callback.save.checkpoint(args$model_prefix)
}

# data
data.shape <- c(28,28,3)
data <- get_iterator(data.shape = data.shape)
train <- data$train
val <- data$value


# train
args$gpus <- c("0")
if (is.null(args$gpus)) {
  print("Computing with CPU")
  devs <- mx.cpu()  
} else {
  print(paste0("GPU option: ", args$gpus))
  devs <- lapply(unlist(strsplit(args$gpus, ",")), function(i) {
    mx.gpu(as.integer(i))
  })
}

#train
time_init <- Sys.time()
model = mx.model.FeedForward.create(
  X                  = train,
  eval.data          = val,
  ctx                = devs,
  symbol             = net,
  eval.metric        = mx.metric.accuracy,
  num.round          = args$num_round,
  learning.rate      = args$lr,
  momentum           = 0.9,
  wd                 = 0.00001,
  kvstore            = args$kv_store,
  array.batch.size   = args$batch_size,
  epoch.end.callback = checkpoint,
  batch.end.callback = mx.callback.log.train.metric(50)
)
time_end <- Sys.time()
difftime(time_end, time_init, units = "mins")

