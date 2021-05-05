import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from model_criticism_mmd.models import TrainedMmdParameters

try:
    import shogun as sg
except ImportError:  # new versions just call it shogun
    raise ImportError('shogun package is unavailable. Thus, we skip 2-sample test with MMD.')


def rbf_mmd_test(x: np.ndarray,
                 y: np.ndarray,
                 trained_params: TrainedMmdParameters,
                 bandwidth: str = 'trained',
                 null_samples: int = 1000,
                 median_samples: int = 1000,
                 cache_size: int = 32):
    '''
    Run an MMD test using a Gaussian kernel.

    Parameters
    ----------
    X : row-instance feature array

    Y : row-instance feature array

    bandwidth : 'median' or 'trained'
        The bandwidth of the RBF kernel (sigma).
        If 'median', estimates the median pairwise distance in the
        aggregate sample and uses that.

    null_samples : int
        How many times to sample from the null distribution.

    median_samples : int
        How many points to use for estimating the bandwidth.

    Returns
    -------
    p_val : float
        The obtained p value of the test.

    stat : float
        The test statistic.

    null_samples : array of length null_samples
        The samples from the null distribution.

    bandwidth : float
        The used kernel bandwidth
    '''
    assert bandwidth in ('median', 'trained')
    if bandwidth == 'median':
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        Z = np.r_[sub(x, median_samples // 2), sub(y, median_samples // 2)]
        D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = np.median(upper, overwrite_input=True)
        sigma = np.sqrt(kernel_width / 2)
        # sigma = median / sqrt(2); works better, sometimes at least
        del Z, D2, upper
    else:
        sigma = trained_params.sigma
        kernel_width = 2 * sigma**2
    # end if

    # as a function z. z: R^n -> R^n
    x__ = trained_params.scales * x
    y__ = trained_params.scales * y

    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.RealFeatures(x__.T.astype(np.float64)))
    mmd.set_q(sg.RealFeatures(y__.T.astype(np.float64)))
    mmd.set_kernel(sg.GaussianKernel(cache_size, kernel_width))

    mmd.set_num_null_samples(null_samples)
    samps = mmd.sample_null()
    stat = mmd.compute_statistic()

    p_val = np.mean(stat <= samps)
    return p_val, stat, samps, bandwidth
