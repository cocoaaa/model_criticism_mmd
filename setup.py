#! -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import sys

version = '1.0'
name = 'model_criticism_mmd'
short_description = 'A Model criticism on distributions via MMD'
long_description = "Suppose that you have 2 datasets(set of samples). " \
                   "You would like to know how close 2 datasets are and also how difference they are. " \
                   "The `model_criticism_mmd` computes a discrepancy between 2 datasets."

setup(
    author='Kensuke Mitsuzawa',
    author_email='kensuke.mit@gmail.com',
    name = name,
    version=version,
    short_description=short_description,
    long_description=long_description,
    keywords=[],
    license="BSD",
    url = "",
    test_suite='test.test_all.suite',
    install_requires=[],
    tests_require=[],
    packages=find_packages()
)