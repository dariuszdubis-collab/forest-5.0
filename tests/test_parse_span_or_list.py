import argparse
import pytest

from forest5.cli import _parse_span_or_list


def test_parse_span_positive_range():
    assert _parse_span_or_list("1-3") == [1, 2, 3]


def test_parse_span_negative_range():
    assert _parse_span_or_list("-3--1") == [-3, -2, -1]


def test_parse_span_list():
    assert _parse_span_or_list("1,4,7") == [1, 4, 7]


def test_parse_span_step_error():
    with pytest.raises(argparse.ArgumentTypeError, match="Step"):
        _parse_span_or_list("1-5:0")


def test_parse_span_hi_less_than_lo():
    with pytest.raises(argparse.ArgumentTypeError, match="Upper bound < lower"):
        _parse_span_or_list("5-3")
