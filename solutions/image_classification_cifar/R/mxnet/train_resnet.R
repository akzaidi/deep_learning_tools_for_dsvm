require(mxnet)
require(argparse)


get_iterator <- function() {
  get_iterator_impl <- function(args) {
    data_dir = args$data_dir
    data.shape <- c(28,28,3)
    
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
  get_iterator_impl
}

parse_args <- function() {
  parser <- ArgumentParser(description='train an image classifer on CIFAR')
  parser$add_argument('--network', type='character', default='resnet-28-small',
                      choices = c('mlp', 'lenet'),
                      help = 'the cnn to use')
  parser$add_argument('--data-dir', type='character', default='cifar10/',
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
source("train_model.R")
if (args$network == 'resnet') {
  source("symbol_resnet.R")
  net <- get_symbol()
} else {
  source("symbol_resnet-28-small.R")
  net <- get_symbol()
}
train_model.fit(args, net, get_iterator())


