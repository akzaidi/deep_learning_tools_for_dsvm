import mxnet as mx
import argparse
import train_model_full


parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--network', type=str, default='resnet_28_small',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='../../data/cifar10/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=50000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log-file', type=str, default='log.txt', help='the name of log file')
parser.add_argument('--log-dir', type=str, default='../../data', help='directory of the log file')
args = parser.parse_args()



# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(10)

# data
def get_iterator(args, kv):
    data_shape = (3, 28, 28)
    kargvs_train ={}
    kargvs_val = {}
    if args.network == 'resnet_28_small':
        kargvs_train = dict(
            mean_img = args.data_dir + "mean.bin"
        )
        kargvs_val = dict(
            mean_img = args.data_dir + "mean.bin"
        )

    elif args.network == 'resnet':
        kargvs_train = dict(
            # 4 pixel padding
            pad=4,
            # Because we use z-score normalization (implemented as a BatchNorm)
            fill_value=127,
            # Shuffle in each epoch as that in
            # https://github.com/gcr/torch-residual-networks/blob/master/data/cifar-dataset.lua
            shuffle=True,
        )
    else:
        print("ERROR in network name")
        exit(-1)#wrong name

    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank,
        **kargvs_train
    )

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank,
        **kargvs_val
    )

    return (train, val)


# train
train_model_full.fit(args, net, get_iterator)


