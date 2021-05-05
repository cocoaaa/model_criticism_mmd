# model_criticism_mmd

The idea is from [the following paper](https://arxiv.org/abs/1611.04488).
`"Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy"`

The original implementation is from [the repository](https://github.com/djsutherland/opt-mmd).

# Setup

I recommend to use conda in order to setup the environment easily.

```
conda env create -f environment.yml
```

You have a Python environment with Python3.6.

Then,

```
python setup install
```

## Note with Python > 3.7

Possible to use Python > 3.7, however, some modules are not available.

```
model_criticism_mmd/supports/mmd_two_sample_test.py
model_criticism_mmd/backends/backend_theano.py
```

# Examples

TDB

# Tests

```
pytest tests
```

# License

The source codes are under the BSD license.

