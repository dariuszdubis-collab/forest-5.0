from __future__ import annotations


class ForestError(Exception):
    pass


class DataValidationError(ForestError):
    pass


class BacktestConfigError(ForestError):
    pass

