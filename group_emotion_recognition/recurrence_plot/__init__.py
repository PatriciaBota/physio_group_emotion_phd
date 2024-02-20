# -*- coding: utf-8 -*-
"""
biosppy
-------

A toolbox for biosignal processing written in Python.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# compat
from __future__ import absolute_import, division, print_function



# allow lazy loading
from .recurrence_analysis import recurrence_quantification_analysis
from .recurrence import recurrence_matrix
