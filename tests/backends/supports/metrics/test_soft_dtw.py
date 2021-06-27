import torch
import numpy as np
from model_criticism_mmd.supports.metrics.soft_dtw import SoftDTW


def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b, False)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads


def test_profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    if torch.cuda.is_available():
        sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    # end if
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        if torch.cuda.is_available():
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)
        # end if

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        if torch.cuda.is_available():
            # Verify the results
            assert torch.allclose(forward_cpu, forward_gpu.cpu())
            assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)
        # end if

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_cpu += [t_cpu]
            if torch.cuda.is_available():
                times_gpu += [t_gpu]
            # end if

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("\tCPU:     ", avg_cpu)
    print("\tGPU:     ", avg_gpu)
    print("\tSpeedup: ", avg_cpu / avg_gpu)
    print()


def test_get_distance_matrix():
    batch_size, len_x, len_y, dims = 8, 15, 12, 5
    x = torch.rand((batch_size, len_x, dims), requires_grad=True)
    y = torch.rand((batch_size, len_y, dims))

    # Create the "criterion" object
    sdtw = SoftDTW(use_cuda=False, gamma=0.1)
    a = sdtw.forward(x, y, True)
    print()


if __name__ == "__main__":
    torch.manual_seed(1234)
    test_get_distance_matrix()
    test_profile(128, 17, 15, 2, tol_backward=1e-6)
    test_profile(512, 64, 64, 2, tol_backward=1e-4)
    test_profile(512, 256, 256, 2, tol_backward=1e-3)
