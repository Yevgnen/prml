#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .sample import ClassificationSample, RegressionSample
from .mnist import loadMNISTImages, loadMNISTLabels

__all__ = ['regression_sample',
           'ClassificationSample',
           'loadMNISTImages',
           'loadMNISTLabels']
