import awkward as ak
import numpy as np
import pandas as pd
import pytest

import akimbo.pandas

import akimbo_ip
import akimbo_ip.accessor  # registers .ip subaccessor


@pytest.fixture
def one_ip4():
    return akimbo.io.ak_to_series(ak.Array(np.array([0], dtype="uint32")), extract=False)


@pytest.fixture
def one_str4():
    return akimbo.io.ak_to_series(ak.Array(["0.0.0.0"]), extract=False)


@pytest.fixture
def one_strnet4():
    return akimbo.io.ak_to_series(ak.Array(["0.0.0.0/24"]), extract=False)


def test_simple4(one_ip4, one_str4):
    out = one_ip4.ak.ip.is_global4()
    assert not out[0]
    out2 = one_ip4.ak.ip.to_string4()
    assert out2[0] == "0.0.0.0"
    
    out = one_str4.ak.ip.parse_address4()
    assert out[0] == b"\x00\x00\x00\x00"


def test_simple_net4(one_strnet4):
    out = one_strnet4.ak.ip.parse_net4()
    assert out[0] == {"prefix": 24, "address": b"\x00\x00\x00\x00"}
    
    out2 = out.ak.ip.contains4(1)
    assert out2[0] is True
    out2 = out.ak.ip.contains4(b"\x00\x00\x00\x01")
    assert out2[0] is True
    out2 = out.ak.ip.contains4("0.0.0.1")
    assert out2[0] is True
