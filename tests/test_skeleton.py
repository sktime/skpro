#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from skpro.skeleton import fib

__author__ = "Frithjof Gressmann"
__copyright__ = "Frithjof Gressmann"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
