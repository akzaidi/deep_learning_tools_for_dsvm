import mxnet as mx
import numpy as np

class Init(mx.init.Xavier):

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if 'proj' in name and name.endswith('weight'):
            self._init_proj(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_proj(self, _, arr):
        '''Initialization of shortcut of kenel (2, 2)'''
        w = np.zeros(arr.shape, np.float32)
        for i in range(w.shape[1]):
            w[i, i, ...] = 0.25
        arr[:] = w


class Scheduler(mx.lr_scheduler.MultiFactorScheduler):

    def __init__(self, epoch_step, factor, epoch_size):
        super(Scheduler, self).__init__(
            step=[epoch_size * s for s in epoch_step],
            factor=factor
        )


@mx.optimizer.Optimizer.register
class Nesterov(mx.optimizer.SGD):

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom *= self.momentum
            grad += wd * weight
            mom += grad
            grad += self.momentum * mom
            weight += -lr * grad
        else:
            assert self.momentum == 0.0
            weight += -lr * (grad + self.wd * weight)

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (
                # Test accuracy 0.930288, 0.930889 and 0.929587 in last 3 epochs
                # if decay weight, bias, gamma and beta like Torch
                # https://github.com/torch/optim/blob/master/sgd.lua
                # Test accuracy 0.927784, 0.928385 and 0.927784 in last 3 epochs
                # if only decay weight
                # Test accuracy 0.928586, 0.929487 and 0.927985 in last 3 epochs
                # if only decay weight and gamma
                n.endswith('_weight')
                or n.endswith('_bias')
                or n.endswith('_gamma')
                or n.endswith('_beta')
            ) or 'proj' in n or 'zscore' in n:
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
            for k, v in attr.items():
                if k.endswith('_wd_mult'):
                    self.wd_mult[k[:-len('_wd_mult')]] = float(v)
        self.wd_mult.update(args_wd_mult)

    def set_lr_mult(self, args_lr_mult):
        """Set individual learning rate multipler for parameters

        Parameters
        ----------
        args_lr_mult : dict of string/int to float
            set the lr multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.lr_mult = {}
        for n in self.idx2name.values():
            if 'proj' in n or 'zscore' in n:
                self.lr_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
            for k, v in attr.items():
                if k.endswith('_lr_mult'):
                    self.lr_mult[k[:-len('_lr_mult')]] = float(v)
        self.lr_mult.update(args_lr_mult)
