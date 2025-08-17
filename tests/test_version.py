from forest5 import __version__


def test_version():
    assert __version__.startswith("5.")
