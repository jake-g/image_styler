# -*- coding: utf-8 -*-


'''setup.py: setuptools control.'''

from setuptools import setup

with open('README.md', 'rb') as f:
    long_descr = f.read().decode('utf-8')

setup(
    name='neural_image_styler',
    packages=['neural_image_styler'],
    entry_points={
        'console_scripts': ['neural_image_styler = neural_image_styler.neural_image_styler:main']
    },
    version=0.1,
    long_description=long_descr,
    description='apply neural styling to an input image',
    url='http://github.com/jake-g/neural_image_styler',
    author='jake-g',
    author_email='omonoid@gmail.com',
    license='MIT'
)
