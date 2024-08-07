import pandas as pd
import pytest

import akimbo.pandas  # registers .ak on pandas

import akimbo_ip.accessor  # registers .ip subaccessor


def test_simple4():
    s1 = pd.Series([0], dtype="uint32")
    out = s1.ak.ip.is_global4()
    assert out[0] is False
    out2 = s1.ak.ip.to_string4()
    assert out2[0] == "0.0.0.0"
    
    s2 = pd.Series("0.0.0.0")
    out = s2.ak.ip.parse_address4()
    assert out[0] == b"\x00\x00\x00\x00"


def test_simple_net4():
    s = pd.Series(["0.0.0.0/24"])
    out = s.ak.ip.parse_net4()
    assert out[0] == {"prefix": 24, "address": b"\x00\x00\x00\x00"}
    
    out2 = out.ak.ip.contains4(1)
    assert out2[0] is True
    out2 = out.ak.ip.contains4(b"\x00\x00\x00\x01")
    assert out2[0] is True
    out2 = out.ak.ip.contains4("0.0.0.1")
    assert out2[0] is True


@pytest.mark.xfail(message="Optional not implemented")
def test_err():
    s = pd.Series(["not-an-ip"])
    s.ak.ip.parse_address4()


def test_6_out():
    s1 = pd.Series([1], dtype="uint32")
    out = s1.ak.ip.to_ipv6_mapped()
    assert out[0] == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x01'
    
