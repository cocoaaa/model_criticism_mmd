# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['model_criticism_mmd',
 'model_criticism_mmd.backends',
 'model_criticism_mmd.backends.kernels_torch',
 'model_criticism_mmd.models',
 'model_criticism_mmd.supports',
 'model_criticism_mmd.supports.metrics',
 'model_criticism_mmd.supports.metrics.soft_dtw']

package_data = \
{'': ['*']}

install_requires = \
['Cython',
 'dataclasses',
 'gpytorch',
 'h5py',
 'nptyping>=1.4.1,<2.0.0',
 'numba',
 'numpy',
 'pandas',
 'scikit-learn',
 'tabulate',
 'torch',
 'torchaudio',
 'torchvision',
 'tqdm>=4.61.2,<5.0.0']

setup_kwargs = {
    'name': 'model-criticism-mmd',
    'version': '2.4',
    'description': 'A Model criticism on distributions via MMD',
    'long_description': '# model_criticism_mmd\n\nThe code to compute MMD value.\nIn the computation process, you can obtain the ARD weight which represents ARD between 2 distributions. \n\nThe idea is from [the following paper](https://arxiv.org/abs/1611.04488).\n`"Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy"`\n\nThe original implementation is from [the repository](https://github.com/djsutherland/opt-mmd).\n\n# Install\n\nTo install the code as a package,\n\n```\nmake install\n```\n\n## Setup full version\n\nSome codes in the package depend on old python versions. Thus, if you want to install full-functionality of the package,\nyou need to build a Python environment.\n\nYou can build it with a conda command.\n\n```\nconda env create -f environment.yml\nconda activate conda_env\n```\n\nYou have a Python environment with Python3.6.\n\nThen,\n\n```\npip install https://github.com/Lasagne/Lasagne/archive/master.zip\npip install https://github.com/Theano/Theano/archive/master.zip\nmake install\n```\n\n\n### Note: Python > 3.7\n\nIf Python > 3.7, some modules are not available.\n\n```\nmodel_criticism_mmd/supports/mmd_two_sample_test.py\nmodel_criticism_mmd/backends/backend_theano.py\n```\n\n# Examples\n\nSee jupyter notebooks. Notes are in `samples/`\n\n# Tests\n\n```\nmake test\n# if GPU machine\n# make test-gpu\n```\n\n# License\n\nThe source code is licensed MIT. The website content is licensed CC BY 4.0.\n\n```\n@misc{sumo-docker-pipeline,\n  author = {Kensuke Mitsuzawa},\n  title = {model-criticism-mmd},\n  year = {2021},\n  publisher = {GitHub},\n  journal = {GitHub repository},\n  howpublished = {\\url{https://github.com/Kensuke-Mitsuzawa/model_criticism_mmd}},\n}\n```\n\n\n',
    'author': 'Kensuke Mitsuzawa',
    'author_email': 'kensuke.mit@gmail.com',
    'maintainer': 'Kensuke Mitsuzawa',
    'maintainer_email': 'kensuke.mit@gmail.com',
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6.1,<4.0.0',
}


setup(**setup_kwargs)
