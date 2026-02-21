#!/usr/bin/env python3
"""Run the tool router model test suite.

Usage:
    python tests.py           # run all tests
    python tests.py -v        # verbose
    python tests.py -k sim    # filter by keyword
"""

import sys
from pathlib import Path

import pytest

sys.exit(pytest.main([str(Path(__file__).parent / "tests"), "-v"] + sys.argv[1:]))
