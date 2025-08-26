import argparse
import pytest

from forest5.utils.argparse_ext import span_or_list

def test_int_positive_range():
    assert span_or_list('1-3') == [1, 2, 3]

def test_int_negative_range():
    assert span_or_list('-3--1') == [-3, -2, -1]

def test_int_list():
    assert span_or_list('1,4,7') == [1, 4, 7]

def test_int_colon_syntax():
    assert span_or_list('8:16:1') == [8, 9, 10, 11, 12, 13, 14, 15, 16]

def test_float_range_colon():
    assert span_or_list('1:2:0.5') == [1.0, 1.5, 2.0]

def test_float_range_dash():
    assert span_or_list('1.0-2.0:0.5') == [1.0, 1.5, 2.0]

def test_float_range_non_divisible():
    assert span_or_list('0:1:0.3') == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.0])

def test_step_error():
    with pytest.raises(argparse.ArgumentTypeError):
        span_or_list('1-5:0')

def test_hi_less_than_lo():
    with pytest.raises(argparse.ArgumentTypeError):
        span_or_list('5-3')
