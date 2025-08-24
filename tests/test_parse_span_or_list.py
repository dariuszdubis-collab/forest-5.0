import argparse
import pytest

from forest5.utils.argparse_ext import span_or_list


def test_parse_span_positive_range_int():
    assert span_or_list("1-3", type_fn=int) == [1, 2, 3]


def test_parse_span_negative_range_int():
    assert span_or_list("-3--1", type_fn=int) == [-3, -2, -1]


def test_parse_span_list_int():
    assert span_or_list("1,4,7", type_fn=int) == [1, 4, 7]


def test_parse_span_colon_syntax_int():
    assert span_or_list("8:16:1", type_fn=int) == [8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_parse_span_default_float():
    assert span_or_list("1.0-1.5:0.25") == [1.0, 1.25, 1.5]


def test_parse_span_step_error():
    with pytest.raises(argparse.ArgumentTypeError, match="Step"):
        span_or_list("1-5:0", type_fn=int)


def test_parse_span_hi_less_than_lo():
    with pytest.raises(argparse.ArgumentTypeError, match="Upper bound < lower"):
        span_or_list("5-3", type_fn=int)

