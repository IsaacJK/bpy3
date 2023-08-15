#!/usr/bin/env python

from distutils.core import setup

setup(name='Barnacle',
      version='0.21',
      description='A probabilistic model of RNA conformational space',
      author='Jes Frellsen, Ida Moltke and Martin Thiim',
      author_email='N/A',
      url='https://sourceforge.net/projects/barnacle-rna/',
      license='GPLv3',
      packages=['Barnacle',
                'Barnacle.Mocapy',
                ],
     )
