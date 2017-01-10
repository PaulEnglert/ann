# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ANN',
    version='0.0.1',
    description='Artificial Nerual Network',
    long_description=readme,
    author='Paul Englert',
    author_email='mail@paulenglert.de',
    url='https://github.com/PaulEnglert/ann',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)