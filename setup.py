#!/usr/bin/env python

# Copyright 2021 Ramon Casero 
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


def read_file(path):
    """
    Return the contents of a file
    Arguments:
      - path: path to the file
    Returns:
      - contents of the file
    """
    with open(path, 'r') as file:
        return file.read().rstrip()


setup(name='simple_pytorch_example',
      version='1.0.0',
      description='Train/validate/test neural network classifiers on FashionMNIST',
      long_description=read_file('README.md'),
      long_description_content_type='text/markdown',
      author='Ramón Casero',
      author_email='rcasero@gmail.com',
      url='https://github.com/rcasero/simple_pytorch_example',
      license='Apache License 2.0',
      packages=find_packages(),
      python_requires='>=3.9',
      install_requires=[
          'ConfigArgParse==1.5.3',
          'dill==0.3.4',
          'matplotlib==3.4.3',
          'mkl-fft==1.3.1',
          'mkl-random==1.2.2',
          'mkl-service==2.4.0',
          'olefile==0.46',
          'pandas==1.3.4',
          'ruamel.yaml.clib==0.2.6',
          'scikit-learn==1.0.1',
          'sty==1.0.0rc2',
          'tensorboard==2.7.0',
          'torchaudio==0.10.0',
          'torchinfo==1.5.3',
          'torchtext==0.11.0',
          'torchvision==0.11.1',
        ],
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Environment :: GPU :: NVIDIA CUDA',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
      )
