# model_criticism_mmd

The code to compute MMD value.
In the computation process, you can obtain the ARD weight which represents ARD between 2 distributions. 

The idea is from [the following paper](https://arxiv.org/abs/1611.04488).
`"Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy"`

The original implementation is from [the repository](https://github.com/djsutherland/opt-mmd).

# Install

To install the code as a package,

```
make install
```

## Setup full version

Some codes in the package depend on old python versions. Thus, if you want to install full-functionality of the package,
you need to build a Python environment.

You can build it with a conda command.

```
conda env create -f environment.yml
conda activate conda_env
```

You have a Python environment with Python3.6.

Then,

```
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install https://github.com/Theano/Theano/archive/master.zip
make install
```


### Note: Python > 3.7

If Python > 3.7, some modules are not available.

```
model_criticism_mmd/supports/mmd_two_sample_test.py
model_criticism_mmd/backends/backend_theano.py
```

# Examples

See jupyter notebooks. Notes are in `samples/`

# Tests

```
make test
```

# License

The source codes are under the BSD license.

