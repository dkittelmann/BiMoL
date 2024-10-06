"""Tests for the `BiMoL` package."""

# %% Import
import unittest

import pytest
from BiMoL.preprocessing.freesurfer import foo

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.fixture()
def response():
    """
    Sample pytest fixture.

    See more at: https://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get("https://github.com/shescher/research-project")


# %% Test Functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_foo():
    """Test the foo function."""
    assert foo() is None


class TestBimol(unittest.TestCase):
    """Tests for `BiMoL` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
