from __future__ import division, print_function

import sys
import time
import types
import typing

import nptyping
import numpy as np
import lasagne
from six import exec_
from sklearn.metrics.pairwise import euclidean_distances
import theano
import theano.tensor as T
from lasagne.layers.base import Layer
from lasagne import init
from nptyping import NDArray
from typing import Any

from model_criticism_mmd.models import TrainedMmdParameters, TrainerBase, TrainingLog
from model_criticism_mmd.logger_unit import logger


floatX = np.dtype(theano.config.floatX)
_eps = 1e-8



def make_floatX(x):
    return np.array(x, dtype=floatX)[()]


class RBFLayer(Layer):
    '''
    An RBF network layer; output the RBF kernel value from each input to a set
    of (learned) centers.
    '''
    def __init__(self, incoming, num_centers,
                 locs=init.Normal(std=1), log_sigma=init.Constant(0.),
                 **kwargs):
        super(RBFLayer, self).__init__(incoming, **kwargs)
        self.num_centers = num_centers

        assert len(self.input_shape) == 2
        in_dim = self.input_shape[1]
        self.locs = self.add_param(locs, (num_centers, in_dim), name='locs',
                                   regularizable=False)
        self.log_sigma = self.add_param(log_sigma, (), name='log_sigma')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_centers)

    def get_output_for(self, input, **kwargs):
        gamma = 1 / (2 * T.exp(2 * self.log_sigma))

        XX = T.dot(input, input.T)
        XY = T.dot(input, self.locs.T)
        YY = T.dot(self.locs, self.locs.T)  # cache this somehow?

        X_sqnorms = T.diagonal(XX)
        Y_sqnorms = T.diagonal(YY)
        return T.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))


class SmoothedCFLayer(Layer):
    '''
    Gets the smoothed characteristic fucntion of inputs, as in eqn (14) of
    Chwialkowski et al. (NIPS 2015).

    Scales the inputs down by sigma, then tests for differences in the
    characteristic functions at locations freqs, smoothed by a Gaussian kernel
    with unit bandwidth.

    NOTE: It's *very* easy for this to get stuck with a bad log_sigma. You
    probably want to initialize it at log(median distance between inputs) or
    similar.
    '''
    def __init__(self, incoming, num_freqs,
                 freqs=init.Normal(std=1), log_sigma=init.Constant(0.),
                 **kwargs):
        super(SmoothedCFLayer, self).__init__(incoming, **kwargs)
        self.num_freqs = num_freqs

        assert len(self.input_shape) == 2
        in_dim = self.input_shape[1]
        self.freqs = self.add_param(freqs, (num_freqs, in_dim), name='freqs')
        self.log_sigma = self.add_param(log_sigma, (), name='log_sigma')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 2 * self.num_freqs)

    def get_output_for(self, input, **kwargs):
        X = input / T.exp(self.log_sigma)
        f = T.exp(-.5 * T.sum(X ** 2, axis=1))[:, np.newaxis]
        angles = T.dot(X, self.freqs.T)
        return T.concatenate([T.sin(angles) * f, T.cos(angles) * f], axis=1)


class ModelTrainerTheanoBackend(TrainerBase):
    def __init__(self):
        self.net_versions = {
            'nothing': self.net_nothing,
            'scaling': self.net_scaling,
            'scaling-exp': self.net_scaling_exp,
            'rbf': self.net_rbf,
            'scf': self.net_scf,
            'basic': self.net_basic}

        self.params = None
        self.train_fn = None
        self.val_fn = None
        self.get_rep = None
        self.log_sigma = None

    @staticmethod
    def net_nothing(net_p, net_q):
        return net_p, net_q, 0

    @staticmethod
    def net_scaling(net_p, net_q):
        net_p = lasagne.layers.ScaleLayer(net_p)
        net_q = lasagne.layers.ScaleLayer(net_q, scales=net_p.scales)
        return net_p, net_q, 0

    @staticmethod
    def net_scaling_exp(net_p, net_q):
        log_scales = theano.shared(np.zeros(net_p.output_shape[1], floatX),
                                   name='log_scales')
        net_p = lasagne.layers.ScaleLayer(net_p, scales=T.exp(log_scales))
        net_q = lasagne.layers.ScaleLayer(net_q, scales=net_p.scales)
        return net_p, net_q, 0

    @staticmethod
    def net_rbf(net_p, net_q, J=5):
        '''
        Network equivalent to Wittawat's mean embedding test:
        compute RBF kernel values to each of J test points.
        '''
        net_p = RBFLayer(net_p, J)
        net_q = RBFLayer(net_q, J, locs=net_p.locs, log_sigma=net_p.log_sigma)
        return net_p, net_q, 0

    @staticmethod
    def net_scf(net_p, net_q, n_freqs=5):
        '''
        Network equivalent to Wittawat's smoothed characteristic function test.
        '''
        net_p = SmoothedCFLayer(net_p, n_freqs)
        net_q = SmoothedCFLayer(net_q, n_freqs,
                                freqs=net_p.freqs, log_sigma=net_p.log_sigma)
        return net_p, net_q, 0

    @staticmethod
    def _paired_dense(in_1, in_2, **kwargs):
        d_1 = lasagne.layers.DenseLayer(in_1, **kwargs)
        d_2 = lasagne.layers.DenseLayer(in_2, W=d_1.W, b=d_1.b, **kwargs)
        return d_1, d_2

    def net_basic(self, net_p, net_q):
        net_p, net_q = self._paired_dense(
            net_p, net_q, num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)
        net_p, net_q = self._paired_dense(
            net_p, net_q, num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify)
        return net_p, net_q, 0

    def register_custom_net(self, code):
        module = types.ModuleType('net_custom', 'Custom network function')
        exec_(code, module.__dict__)
        sys.modules['net_custom'] = module
        self.net_versions['custom'] = module.net_custom

    ################################################################################
    ### MMD Eqs.

    def rbf_mmd2_and_ratio(self,
                           X: theano.tensor.TensorVariable,
                           Y: theano.tensor.TensorVariable,
                           sigma: float=0,
                           biased=True):
        gamma = 1 / (2 * sigma ** 2)

        XX = T.dot(X, X.T)
        XY = T.dot(X, Y.T)
        YY = T.dot(Y, Y.T)

        X_sqnorms = T.diagonal(XX)
        Y_sqnorms = T.diagonal(YY)
        # todo is it a kernel function including ARD weights ??
        K_XY = T.exp(-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = T.exp(-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = T.exp(-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

        return self._mmd2_and_ratio(K_XX, K_XY, K_YY, unit_diagonal=True, biased=biased)

    def _mmd2_and_ratio(self,
                        K_XX, K_XY, K_YY, unit_diagonal=False, biased=False,
                        min_var_est=_eps) -> typing.Tuple[theano.tensor.Elemwise, theano.tensor.Elemwise]:
        """compute mmd^2 value and ratio(corresponding to t-value)

        :return: (mmd^2, t-value)
        """
        mmd2, var_est = self._mmd2_and_variance(
            K_XX, K_XY, K_YY, unit_diagonal=unit_diagonal, biased=biased)
        ratio = mmd2 / T.sqrt(T.largest(var_est, min_var_est))
        return mmd2, ratio

    def _mmd2_and_variance(self, K_XX, K_XY, K_YY, unit_diagonal=False, biased=False):
        m = K_XX.shape[0]  # Assumes X, Y are same shape

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if unit_diagonal:
            diag_X = diag_Y = 1
            sum_diag_X = sum_diag_Y = m
            sum_diag2_X = sum_diag2_Y = m
        else:
            diag_X = T.diagonal(K_XX)
            diag_Y = T.diagonal(K_YY)

            sum_diag_X = diag_X.sum()
            sum_diag_Y = diag_Y.sum()

            sum_diag2_X = diag_X.dot(diag_X)
            sum_diag2_Y = diag_Y.dot(diag_Y)

        Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(axis=0)
        K_XY_sums_1 = K_XY.sum(axis=1)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        # TODO: turn these into dot products?
        # should figure out if that's faster or not on GPU / with theano...
        Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
        Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
        K_XY_2_sum = (K_XY ** 2).sum()

        if biased:
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2 * K_XY_sum / (m * m))
        else:
            mmd2 = (Kt_XX_sum / (m * (m - 1))
                    + Kt_YY_sum / (m * (m - 1))
                    - 2 * K_XY_sum / (m * m))

        var_est = (
                2 / (m ** 2 * (m - 1) ** 2) * (
                2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
                + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
                - (4 * m - 6) / (m ** 3 * (m - 1) ** 3) * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
                + 4 * (m - 2) / (m ** 3 * (m - 1) ** 2) * (
                        K_XY_sums_1.dot(K_XY_sums_1)
                        + K_XY_sums_0.dot(K_XY_sums_0))
                - 4 * (m - 3) / (m ** 3 * (m - 1) ** 2) * K_XY_2_sum
                - (8 * m - 12) / (m ** 5 * (m - 1)) * K_XY_sum ** 2
                + 8 / (m ** 3 * (m - 1)) * (
                        1 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
                        - Kt_XX_sums.dot(K_XY_sums_1)
                        - Kt_YY_sums.dot(K_XY_sums_0))
        )

        return mmd2, var_est

    ################################################################################
    ### Adding loss and so on to the network

    def make_network(self,
                     input_p: theano.tensor.TensorVariable,
                     input_q: theano.tensor.TensorVariable,
                     dim: int,
                     criterion='ratio', biased=True, streaming_est=False,
                     linear_kernel=False, log_sigma: float=0, hotelling_reg=0,
                     opt_log=True, batchsize=None,
                     net_version='nothing'):

        # --------------------------
        # definition of layers which convert input-Data(X, Y) into weighted-value.
        # The operation corresponds with 'z' function.
        in_p = lasagne.layers.InputLayer(shape=(batchsize, dim), input_var=input_p)
        in_q = lasagne.layers.InputLayer(shape=(batchsize, dim), input_var=input_q)
        # note: net_versions is global variable. A dict object. A key is string, A value is function.
        # in_p, in_q are arguments into functions.
        # net_p, net_q, reg = net_versions[net_version](in_p, in_q)
        # a layer to scale the in_p and in_q.
        # scales variable corresponds with 'z' function in the paper.
        net_p, net_q, reg = self.net_scaling(in_p, in_q)
        rep_p, rep_q = lasagne.layers.get_output([net_p, net_q])
        # --------------------------

        # definition of MMD (in variations)
        # choices = {  # criterion, linear kernel, streaming
        #     ('mmd', False, False): mmd.rbf_mmd2,
        #     ('mmd', False, True): mmd.rbf_mmd2_streaming,
        #     ('mmd', True, False): mmd.linear_mmd2,
        #     ('ratio', False, False): mmd.rbf_mmd2_and_ratio,
        #     ('ratio', False, True): mmd.rbf_mmd2_streaming_and_ratio,
        #     ('ratio', True, False): mmd.linear_mmd2_and_ratio,
        #     ('hotelling', True, False): mmd.linear_mmd2_and_hotelling,
        # }
        # try:
        #     fn = choices[criterion, linear_kernel, streaming_est]
        # except KeyError:
        #     raise ValueError("Bad parameter combo: criterion = {}, {}, {}".format(
        #         criterion,
        #         "linear kernel" if linear_kernel else "rbf kernel",
        #         "streaming" if streaming_est else "not streaming"))

        kwargs = {}
        if linear_kernel:
            log_sigma = None
        else:
            log_sigma = theano.shared(make_floatX(log_sigma), name='log_sigma')
            kwargs['sigma'] = T.exp(log_sigma)
        if not streaming_est:
            kwargs['biased'] = biased
        if criterion == 'hotelling':
            kwargs['reg'] = hotelling_reg

        mmd2_pq, stat = self.rbf_mmd2_and_ratio(X=rep_p, Y=rep_q, **kwargs)
        # obj is "objective" value.
        obj = -(T.log(T.largest(stat, 1e-6)) if opt_log else stat) + reg
        return mmd2_pq, obj, rep_p, net_p, net_q, log_sigma

    def setup(self,
              dim,
              criterion='ratio',
              biased=True,
              streaming_est=False,
              opt_log=True,
              linear_kernel=False,
              opt_sigma=False,
              init_log_sigma: float=0,
              net_version='basic',
              hotelling_reg=0,
              strat='nesterov_momentum',
              learning_rate=0.01, **opt_args):
        """

        :param dim:
        :param criterion:
        :param biased:
        :param streaming_est:
        :param opt_log:
        :param linear_kernel:
        :param opt_sigma:
        :param init_log_sigma:
        :param net_version:
        :param hotelling_reg:
        :param strat:
        :param learning_rate:
        :param opt_args:
        :return:
        """
        # definition of input variable.
        input_p = T.matrix('input_p')
        input_q = T.matrix('input_q')

        mmd2_pq, obj, rep_p, net_p, net_q, log_sigma = self.make_network(
            input_p, input_q, dim,
            criterion=criterion, biased=biased, streaming_est=streaming_est,
            opt_log=opt_log, linear_kernel=linear_kernel, log_sigma=init_log_sigma,
            hotelling_reg=hotelling_reg, net_version=net_version)
        # Returns a list of Theano shared variables or expressions that parameterize the layer.
        params: typing.List[theano.tensor.sharedvar.TensorSharedVariable] = \
            lasagne.layers.get_all_params([net_p, net_q], trainable=True)
        sacles: theano.tensor.sharedvar.TensorSharedVariable = [p for p in params if p.name == 'scales'][0]
        if opt_sigma:
            params.append(log_sigma)
        # end if

        # definition of gradient-search.
        # generate a function-object which can take arguments.
        fn = getattr(lasagne.updates, strat)
        # updates(return of lasagne.updates) is a dictionary-obj. The dict-obj is with keys:
        updates: typing.Dict[theano.tensor.sharedvar.TensorSharedVariable, theano.tensor.var.TensorVariable] = \
            fn(obj, params, learning_rate=learning_rate, **opt_args)

        logger.info("Compiling...")
        # a function for training. updates,
        # updates is key-value objects. The key a name of variable, the value is a way to update the variable.
        train_fn = theano.function(
            inputs=[input_p, input_q], outputs=[mmd2_pq, obj], updates=updates)
        val_fn = theano.function(inputs=[input_p, input_q], outputs=[mmd2_pq, obj])
        get_rep = theano.function(inputs=[input_p], outputs=rep_p)
        logger.info("done")

        return params, train_fn, val_fn, get_rep, log_sigma, sacles


    ################################################################################
    ### Training helpers

    @staticmethod
    def iterate_minibatches(*arrays, **kwds):
        batchsize = kwds['batchsize']
        shuffle = kwds.get('shuffle', False)

        assert len(arrays) > 0
        n = len(arrays[0])
        assert all(len(a) == n for a in arrays[1:])

        if shuffle:
            indices = np.arange(n)
            np.random.shuffle(indices)

        for start_idx in range(0, max(0, n - batchsize) + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield tuple(a[excerpt] for a in arrays)

    def run_train_epoch(self, X_train, Y_train, batchsize: int, train_fn, log_sigma, scales, is_shuffle: bool = True) -> typing.Tuple[float, float]:
        total_mmd2 = 0
        total_obj = 0
        n_batches = 0
        batches = zip(  # shuffle the two independently
            self.iterate_minibatches(X_train, batchsize=batchsize, shuffle=is_shuffle),
            self.iterate_minibatches(Y_train, batchsize=batchsize, shuffle=is_shuffle),
        )
        for ((Xbatch,), (Ybatch,)) in batches:
            mmd2, obj = train_fn(Xbatch, Ybatch)
            assert np.isfinite(mmd2)
            assert np.isfinite(obj)
            total_mmd2 += mmd2
            total_obj += obj
            n_batches += 1
            # logger.info(log_sigma.eval(), scales.eval()) TODO delete
        return total_mmd2 / n_batches, total_obj / n_batches

    def run_val(self, X_val, Y_val, batchsize, val_fn) -> typing.Tuple[float, float]:
        total_mmd2 = 0
        total_obj = 0
        n_batches = 0
        for (Xbatch, Ybatch) in self.iterate_minibatches(
                X_val, Y_val, batchsize=batchsize):
            mmd2, obj = val_fn(Xbatch, Ybatch)
            assert np.isfinite(mmd2)
            assert np.isfinite(obj)
            total_mmd2 += mmd2
            total_obj += obj
            n_batches += 1
        # end for
        return total_mmd2 / n_batches, total_obj / n_batches

    def __train(self,
                X_train: NDArray[(Any, Any), Any],
                Y_train: NDArray[(Any, Any), Any],
                X_val: NDArray[(Any, Any), Any],
                Y_val: NDArray[(Any, Any), Any],
                criterion='ratio',
                biased=True,
                streaming_est=False,
                opt_log=True,
                linear_kernel=False,
                hotelling_reg=0,
                init_log_sigma: float=0.0,
                opt_sigma=False,
                init_sigma_median=False,
                num_epochs=10000,
                batchsize=200,
                val_batchsize=1000,
                verbose=True,
                net_version='basic',
                opt_strat='nesterov_momentum',
                learning_rate: float=0.01,
                log_params: bool=False,
                init_scales: typing.Optional[np.ndarray] = None, **opt_args):
        # assert in order to check objects are 2nd order Tensor.
        assert X_train.ndim == X_val.ndim == Y_train.ndim == Y_val.ndim == 2
        dim = X_train.shape[1]
        assert X_val.shape[1] == Y_train.shape[1] == Y_val.shape[1] == dim

        if linear_kernel:
            logger.info("Using linear kernel")
        elif opt_sigma:
            logger.info("Starting with sigma = {}; optimizing it".format(
                'median' if init_sigma_median else np.exp(init_log_sigma)))
        else:
            logger.info("Using sigma = {}".format(
                'median' if init_sigma_median else np.exp(init_log_sigma)))
        # definition of graph
        params, train_fn, val_fn, get_rep, log_sigma, scales = self.setup(
            dim, criterion=criterion, linear_kernel=linear_kernel,
            biased=biased, streaming_est=streaming_est,
            hotelling_reg=hotelling_reg,
            init_log_sigma=init_log_sigma,
            opt_sigma=opt_sigma,
            opt_log=opt_log, net_version=net_version,
            strat=opt_strat, learning_rate=learning_rate, **opt_args)

        # initialization of initial-sigma value
        if log_sigma is not None and init_sigma_median:
            logger.info("Getting median initial sigma value...")
            n_samp = min(500, X_train.shape[0], Y_train.shape[0])
            samp = np.vstack([
                X_train[np.random.choice(X_train.shape[0], n_samp, replace=False)],
                Y_train[np.random.choice(Y_train.shape[0], n_samp, replace=False)],
            ])
            reps = np.vstack([
                get_rep(batch) for batch, in
                self.iterate_minibatches(samp, batchsize=val_batchsize)])
            D2 = euclidean_distances(reps, squared=True)
            med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
            log_sigma.set_value(make_floatX(np.log(med_sqdist / np.sqrt(2)) / 2))
            rep_dim = reps.shape[1]
            del samp, reps, D2, med_sqdist
            logger.info("{:.3g}".format(np.exp(log_sigma.get_value())))
        else:
            rep_dim = get_rep(X_train[:1]).shape[1]
        # end if

        if init_scales is not None:
            logger.info('Set initial scales-value.')
            assert X_train.shape[1] == Y_train.shape[1] == init_scales.shape[0]
            scales.set_value(init_scales)
        # end if

        logger.info("Input dim {}, representation dim {}".format(
            X_train.shape[1], rep_dim))
        logger.info("Training on {} samples (batch {}), validation on {} (batch {})"
              .format(X_train.shape[0], batchsize, X_val.shape[0], val_batchsize))
        logger.info("{} parameters to optimize: {}".format(
            len(params), ', '.join(p.name for p in params)))

        # ndarray for log message
        # value_log = np.zeros(num_epochs + 1, dtype=[
        #                                                ('train_mmd', floatX), ('train_obj', floatX),
        #                                                ('val_mmd', floatX), ('val_obj', floatX),
        #                                                ('elapsed_time', np.float64)]
        #                                            + ([('sigma', floatX)] if opt_sigma else [])
        #                                            + ([('params', object)] if log_params else []))
        value_log = []
        # format of log output
        fmt = ("{: >6,}: avg train MMD^2 {: .6f} obj {: .6f},  "
               "avg val MMD^2 {: .6f}  obj {: .6f}  elapsed: {:,}s")
        if opt_sigma:
            fmt += '  sigma: {sigma:.3g}'
        # end if

        def log(epoch, t_mmd2, t_obj, v_mmd2, v_job, t):
            sigma = np.exp(float(params[-1].get_value())) if opt_sigma else None
            if verbose and (epoch in {0, 5, 25, 50}
                            # or (epoch < 1000 and epoch % 50 == 0)
                            or epoch % 100 == 0):
                logger.info(fmt.format(epoch, t_mmd2, t_obj, v_mmd2, v_obj, int(t), sigma=sigma))
            # end if
            tup = (t_mmd2, t_obj, v_mmd2, v_obj, t)
            if opt_sigma:
                tup += (sigma,)
            # end if
            if log_params:
                tup += ([p.get_value() for p in params],)
            # end if
            log_obj = TrainingLog(epoch, t_mmd2, t_obj, v_mmd2, t_mmd2, sigma, scales.get_value())
            value_log.append(log_obj)
        # end def

        t_mmd2, t_obj = self.run_val(X_train, Y_train, batchsize, val_fn)
        v_mmd2, v_obj = self.run_val(X_val, Y_val, val_batchsize, val_fn)
        log(0, t_mmd2, t_obj, v_mmd2, v_obj, 0)
        start_time = time.time()
        # logger.info(f'{0},{t_mmd2},{t_obj},{scales.get_value()},{log_sigma.get_value()}')
        for epoch in range(1, num_epochs + 1):
            try:
                t_mmd2, t_obj = self.run_train_epoch(
                    X_train, Y_train, batchsize, train_fn, log_sigma, scales, is_shuffle=True)
                v_mmd2, v_obj = self.run_val(X_val, Y_val, val_batchsize, val_fn)
                log(epoch, t_mmd2, t_obj, v_mmd2, v_obj, time.time() - start_time)

            except KeyboardInterrupt:
                break
            # end try
        # end for
        sigma = np.exp(log_sigma.get_value()) if log_sigma is not None else None
        return ([p.get_value() for p in params], [p.name for p in params],
                get_rep, value_log, sigma)

    # interface
    def train(self,
              x: nptyping.NDArray[(typing.Any, typing.Any), typing.Any],
              y: nptyping.NDArray[(typing.Any, typing.Any), typing.Any],
              num_epochs: int = 500,
              batchsize: int = 200,
              val_batchsize: int = 1000,
              ratio_train: float = 0.8,
              init_scales: nptyping.NDArray[(typing.Any,), typing.Any] = None,
              init_sigma_median: bool = False,
              opt_sigma: bool = True,
              opt_log: bool = True,
              opt_strategy: str = 'nesterov_momentum',
              x_val=None,
              y_val=None,
              init_log_sigma: float = 0.0) -> TrainedMmdParameters:
        '''optimize parameters based on t-value.

        :param x: training data
        :param y: training data
        :param num_epochs: epoch size
        :param batchsize: batch size
        :param val_batchsize: batch size for validation
        :param ratio_train: a ratio to split data into (train, val). Not used when x_val, y_val are given.
        :param init_scales: initial value for a scales vector. The dimension size should be same as the one of data.
        :param init_sigma_median: initialize sigma parameter with median.
        :param opt_sigma: If True, then optimization of sigma too, False not.
        :param opt_log: If True, an objective value is -(T.log(T.largest(stat, 1e-6)) + reg; False stat + reg.
        :param opt_strategy: a key name of opt-strategies. See a doc. of Lasagne. https://lasagne.readthedocs.io/en/latest/modules/updates.html
        :param x_val: validation data corresponding to x.
        :param y_val: validation data corresponding to y.
        :param init_log_sigma: an initial value for sigma.
        :return:
        '''
        assert len(x) == len(y), 'currently, len(x) and len(y) must be same.'
        if init_scales is not None:
            assert init_scales.shape[-1] == x.shape[1], \
                f'unmatched dim. size. data has {x.shape[1]}, init_scales has {init_scales.shape[0]}'
        # end if

        if x_val is None or y_val is None:
            n_train = int(len(x) * ratio_train)
            np.random.shuffle(x)
            np.random.shuffle(y)

            x_train = x[:n_train]
            x_val = x[n_train:]
            y_train = y[:n_train]
            y_val = y[n_train:]
        else:
            x_train = x
            y_train = y
            x_val = x_val
            y_val = y_val
        # end if

        params, param_names, get_rep, value_log, sigma = self.__train(
            X_train=x_train,
            Y_train=y_train,
            X_val=x_val,
            Y_val=y_val,
            criterion='ratio',
            biased=True,
            hotelling_reg=0,
            init_log_sigma=init_log_sigma,
            opt_sigma=opt_sigma,
            opt_log=opt_log,
            num_epochs=num_epochs,
            batchsize=batchsize,
            val_batchsize=val_batchsize,
            init_sigma_median=init_sigma_median,
            init_scales=init_scales,
            opt_strat=opt_strategy,
            net_version='scaling')
        self.get_rep = get_rep
        logger.info(f'Trained result: the opt global-sigma: {sigma}')
        result = TrainedMmdParameters(
            sigma=sigma,
            scales=params[0],
            training_log=value_log,
            x_train=x_train,
            y_train=y_train,
            func_mapping_network=self.get_rep)

        return result

    # ------------------------------------------------------------------------------------------------
    # compute MMD value

    def mmd_distance(self,
                     x: nptyping.NDArray[(typing.Any, typing.Any), typing.Any],
                     y: nptyping.NDArray[(typing.Any, typing.Any), typing.Any],
                     sigma: float) -> typing.Tuple[float, float]:
        # note: no need to give scale-vector because self.get_rep() is a network to map: X -> scaled_X
        assert self.get_rep is not None, 'run train() first'
        x_rep = self.get_rep(x)
        y_rep = self.get_rep(y)
        __mmd2, __ratio = self.rbf_mmd2_and_ratio(X=x_rep, Y=y_rep, sigma=sigma)
        mmd2 = __mmd2.eval()
        ratio = __ratio.eval()
        return mmd2, ratio
