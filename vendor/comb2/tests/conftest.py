from __future__ import annotations


import pytest


@pytest.fixture(scope="session")
def cpu_float8_supported() -> bool:
    from comb2.codec import probe_cpu_float8_cast

    return probe_cpu_float8_cast()
