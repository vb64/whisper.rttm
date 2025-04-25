"""Module init.py tests.

make test T=test_init.py
"""
import torch

from . import TestBase


class TestInit(TestBase):
    """Module __init__.py."""

    def test_torch_types(self):
        """Check torch types."""
        # print(repr(torch.float16))
        assert torch.float16
