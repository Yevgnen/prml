#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='mlpy',
    version='1.0',
    packages=find_packages('src'),
    package_dir = {'': 'src'}
)
