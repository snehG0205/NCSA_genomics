#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:32:16 2020

@author: root
"""

from setuptools import setup
setup(name='TSForecasting',
version='0.1',
description='Package for TS imputations using LSTM',
url='#',
author='Sneh',
author_email='sneh.tools@gmail.com',
license='MIT',
packages=['TSForecasting'],
include_package_data=True,
 package_data={
        # If any package contains *.txt files, include them:
        "": ["*.txt","*.csv"]},
install_requires=[
          'pandas', 'numpy', 'matplotlib', 'sklearn', 'keras', 'scipy', 'tabulate', 'random', 're', 'dateutil', 'warnings', 'operator', 'enums', 'collections'
      ],
zip_safe=False)