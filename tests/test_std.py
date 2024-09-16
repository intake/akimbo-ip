"""Tests of functionality straight from the rust documentation"""
import pandas as pd
import pytest


def test_broadcast4():
    s = pd.Series(["172.16.0.0/22"]).ak.ip.parse_net4()
    expected = pd.Series("172.16.3.255").ak.ip.parse_address4()
    out = s.ak.ip.broadcast4()
    assert (out == expected).all()


def test_network4():
    s = pd.Series(["172.16.123.123/16"]).ak.ip.parse_net4()
    expected = pd.Series("172.16.0.0").ak.ip.parse_address4()
    out = s.ak.ip.network4()
    assert (out == expected).all()


def test_hostmast4():
    s = pd.Series(["10.1.0.0/20"]).ak.ip.parse_net4()
    expected = pd.Series("0.0.15.255").ak.ip.parse_address4()
    out = s.ak.ip.hostmask4()
    assert (out == expected).all()


def test_netmask4():
    s = pd.Series(["10.1.0.0/20"]).ak.ip.parse_net4()
    expected = pd.Series("255.255.240.0").ak.ip.parse_address4()
    out = s.ak.ip.netmask4()
    assert (out == expected).all()


def test_trunc4():
    s = pd.Series(["192.168.12.34/16"]).ak.ip.parse_net4()
    expected = pd.Series("192.168.0.0/16").ak.ip.parse_net4()
    out = s.ak.ip.trunc4()
    assert out.ak.to_list() == expected.ak.to_list()


def test_supernet4():
    s = pd.Series(["172.16.1.0/24"]).ak.ip.parse_net4()
    expected = pd.Series("172.16.0.0/23").ak.ip.parse_net4()
    out = s.ak.ip.supernet4()
    assert out.ak.to_list() == expected.ak.to_list()

   
def test_subnets4():
    s = pd.Series(["10.0.0.0/24"]).ak.ip.parse_net4()
    expected = pd.Series(
        [["10.0.0.0/26",
          "10.0.0.64/26",
          "10.0.0.128/26",
          "10.0.0.192/26"]],
    ).ak.ip.parse_net4()
    out = s.ak.ip.subnets4(26)
    assert out.ak.to_list() == expected.ak.to_list()
    s = pd.Series(["10.0.0.0/30"]).ak.ip.parse_net4()
    expected = pd.Series(
        [["10.0.0.0/32",
          "10.0.0.1/32",
          "10.0.0.2/32",
          "10.0.0.3/32"]]
    ).ak.ip.parse_net4()
    out = s.ak.ip.subnets4(32)
    assert out.ak.to_list() == expected.ak.to_list()


def test_aggregate4():
    s = pd.Series([[
        "10.0.0.0/24",
        "10.0.1.0/24",
        "10.0.2.0/24"
    ]]).ak.ip.parse_net4()
    expected = pd.Series([[
        "10.0.0.0/23",
        "10.0.2.0/24"
    ]]).ak.ip.parse_net4()
    out = s.ak.ip.aggregate4()
    assert out.ak.to_list() == expected.ak.to_list()
