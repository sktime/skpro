# -*- coding: utf-8 -*-
import pkg_resources

import skpro.parametric
import skpro.workflow

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'
