import pyarrow as pa
import pandas as pd
import pytest

bytestring4 = pd.ArrowDtype(pa.binary(4))
bytestring16 = pd.ArrowDtype(pa.binary(16))


def test_simple4():
    s1 = pd.Series([0], dtype="u4")
    out = s1.ak.ip.is_global4()
    assert out[0] is False
    out2 = s1.ak.ip.to_string4()
    assert out2[0] == "0.0.0.0"
    
    s2 = pd.Series("0.0.0.0")
    out = s2.ak.ip.parse_address4()
    assert out[0] == b"\x00\x00\x00\x00"


def test_simple6():
    s1 = pd.Series([b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01"],
                   dtype=bytestring16)
    out = s1.ak.ip.is_global6()
    assert out.tolist() == [False, False]
    
    out2 = s1.ak.ip.to_string6()
    assert out2.tolist() == ["::", "::1"]
    out3 = out2.ak.ip.parse_address6()
    assert out3[1] == s1[1]


def test_to_lists():
    s1 = pd.Series([b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
                    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01"],
                   dtype=bytestring16)
    out = s1.ak.ip.to_int_list()
    assert out.to_list() == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    out2 = out.ak.ip.to_bytestring()
    assert s1.to_list() == out2.to_list()
        
    s2 = pd.Series([0, 1], dtype="uint32")
    out = s2.ak.ip.to_int_list()
    assert out.to_list() == [[0, 0, 0, 0], [1, 0, 0, 0]]
    out2 = out.ak.ip.to_bytestring()
    assert out2.to_list() == [b'\x00\x00\x00\x00', b'\x01\x00\x00\x00']


def test_nested():
    s = pd.DataFrame({"a": [0], "b": [0]}).ak.merge()
    out = s.ak.ip.is_global4(where="b")
    assert out[0] == {"a": 0, "b": False}


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
    s1 = pd.Series([1], dtype="u4")
    out = s1.ak.ip.to_ipv6_mapped()
    assert out[0] == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\x00\x00\x00\x01'
    

def test_rename():
    s = pd.DataFrame({"address": pd.Series([1], dtype="u4"), 
                      "end": pd.Series([16], dtype="u1")}).ak.merge()
    out = s.ak.ip.contains4(b"\x00\x00\x00\x01")
    assert s.tolist() == out.tolist()  # no change, no match
    out = out.ak.ip.contains4(b"\x00\x00\x00\x01", match_kwargs={"prefix": "end"})
    assert out[0] is True


def test_inner_list_hosts():
    # note: both addresses are rounded down
    s = pd.DataFrame({"address": pd.Series([1, 2], dtype="u4"), 
                      "prefix": pd.Series([31, 29], dtype="u1")}).ak.merge()
    out = s.ak.ip.hosts4()
    assert out.to_list() == [
        # includes gateway/broadcast
        [b'\x00\x00\x00\x00', b'\x01\x00\x00\x00'], 
        # does not include gateway/broadcast
        [b'\x01\x00\x00\x00', b'\x02\x00\x00\x00', b'\x03\x00\x00\x00', b'\x04\x00\x00\x00', b'\x05\x00\x00\x00', b'\x06\x00\x00\x00']
    ]


def test_masks():
    s = pd.Series(["7.7.7.7", "8.8.8.8"]).ak.ip.parse_address4()
    out1 = s.ak.ip | s.ak.array[:1]
    assert out1.ak.ip.to_int_list().tolist() == [[7, 7, 7, 7], [15, 15, 15, 15]]
    
    out2 = s.ak.ip | "255.0.0.0"
    assert out2.ak.ip.to_int_list().tolist() == [[255, 7, 7, 7], [255, 8, 8, 8]]
    