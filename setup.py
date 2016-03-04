#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='mlpy',
    version='0.1',
    author='Yevgnen',
    packages=find_packages('mlpy'),
    package_dir = {'': 'mlpy'},
    zip_safe=False
)
