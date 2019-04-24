from setuptools import find_packages
from setuptools import setup

setup(
  name='trainer',
  version='0.1',
  packages=find_packages(),
  description='trainer package',
  license='MIT',
  install_requires=[
      'fire>=0.1.3',
      'regex==2017.4.5',
      'cloudstorage'
  ],
  zip_safe=False)
