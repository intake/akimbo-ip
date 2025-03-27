import functools
import ipaddress

import akimbo_ip.akimbo_ip as lib
import awkward as ak
import numpy as np
from akimbo.apply_tree import dec
from akimbo.mixin import EagerAccessor, LazyAccessor
from akimbo_ip import utils
from awkward import contents, index


def match_ip4(arr):
    """matches fixed-list[4, u8] and fixed-bytestring[4] and ANY 4-byte value (like uint32, assumed big-endian"""
    if arr.is_leaf:
        return arr.dtype.itemsize == 4
    return (
        arr.is_regular
        and arr.size == 4
        and isinstance(arr.content, ak.contents.Content)
        and arr.content.is_leaf
        and arr.content.dtype.itemsize == 1
    )


def match_ip6(arr):
    """matches fixed-list[16, u8] and fixed-bytestring[16]"""
    return (
        arr.is_regular
        and arr.size == 16
        and arr.content.is_leaf
        and arr.content.dtype.itemsize == 1
    )


def match_ip(arr):
    """matches either v4 or v6 IPs"""
    return match_ip4(arr) or match_ip6(arr)


def match_prefix(arr):
    """A network prefix is always one byte"""
    return arr.is_leaf and arr.dtype.itemsize == 1


def match_net4(arr, address="address", prefix="prefix"):
    """Matches a record with IP4 field and prefix field (u8)"""
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip4(arr[address])
        and match_prefix(arr[prefix])
    )


def match_net6(arr, address="address", prefix="prefix"):
    """Matches a record with IP6 field and prefix field (u8)"""
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip6(arr[address])
        and match_prefix(arr[prefix])
    )


def match_list_net4(arr, address="address", prefix="prefix"):
    """Matches lists of ip4 network records"""
    if arr.is_list:
        cont = arr.content.content if arr.content.is_option else arr.content
        return match_net4(cont)
    return False


def match_stringlike(arr):
    return "string" in arr.parameters.get("__array__", "")


def parse_address4(str_arr):
    """Interpret (byte)strings as IPv4 addresses

    Output will be fixed length 4 bytestring array
    """
    out, valid = lib.parse4(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return contents.ByteMaskedArray(
        index.Index8(valid), utils.u8_to_ip4(out.view("uint8")), True
    )


def parse_address6(str_arr):
    """Interpret (byte)strings as IPv6 addresses

    Output will be fixed length 4 bytestring array
    """
    out = lib.parse6(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return utils.u8_to_ip6(out.view("uint8"))


def parse_net4(str_arr):
    """Interpret (byte)strings as IPv4 networks (address/prefix)

    Output will be a record array {"address": fixed length 4 bytestring, "prefix": uint8}
    """
    out = lib.parsenet4(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return contents.RecordArray(
        [
            contents.RegularArray(
                contents.NumpyArray(
                    out[0].view("uint8"), parameters={"__array__": "byte"}
                ),
                size=4,
                parameters={"__array__": "bytestring"},
            ),
            contents.NumpyArray(out[1]),
        ],
        fields=["address", "prefix"],
    )


def contains4(nets, other, address="address", prefix="prefix"):
    # TODO: this is single-value only
    arr = nets[address]
    if arr.is_leaf:
        arr = arr.data.astype("uint32")
    else:
        # fixed bytestring or 4 * uint8 regular
        arr = arr.content.data.view("uint32")
    ip = ipaddress.IPv4Address(other)._ip
    out = lib.contains_one4(arr, nets[prefix].data.astype("uint8"), ip)
    return contents.NumpyArray(out)


def hosts4(nets, address="address", prefix="prefix"):
    (arr,) = to_ip4(nets[address])
    ips, offsets = lib.hosts4(arr, nets[prefix].data.astype("uint8"))
    return contents.ListOffsetArray(index.Index64(offsets), utils.u8_to_ip4(ips))


def network4(nets, address="address", prefix="prefix"):
    (arr,) = to_ip4(nets[address])
    out = lib.network4(arr, nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)


def broadcast4(nets, address="address", prefix="prefix"):
    (arr,) = to_ip4(nets[address])
    out = lib.broadcast4(arr, nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)


def hostmask4(nets, address="address", prefix="prefix"):
    out = lib.hostmask4(nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)


def netmask4(nets, address="address", prefix="prefix"):
    out = lib.netmask4(nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)


def trunc4(nets, address="address", prefix="prefix"):
    (arr,) = to_ip4(nets[address])
    out = lib.trunc4(arr, nets[prefix].data.astype("uint8"))
    return contents.RecordArray(
        [utils.u8_to_ip4(out), nets[prefix]], fields=[address, prefix]
    )


def supernet4(nets, address="address", prefix="prefix"):
    (arr,) = to_ip4(nets[address])
    out = lib.supernet4(arr, nets[prefix].data.astype("uint8"))
    return contents.RecordArray(
        [utils.u8_to_ip4(out), contents.NumpyArray(nets[prefix].data - 1)],
        fields=[address, prefix],
    )


def subnets4(nets, new_prefix, address="address", prefix="prefix"):
    (arr,) = to_ip4(nets[address])
    out, offsets = lib.subnets4(arr, nets[prefix].data.astype("uint8"), new_prefix)
    addr = utils.u8_to_ip4(out)
    return contents.ListOffsetArray(
        index.Index64(offsets),
        contents.RecordArray(
            [
                addr,
                contents.NumpyArray(np.full((len(addr),), new_prefix, dtype="uint8")),
            ],
            fields=[address, prefix],
        ),
    )


def aggregate4(net_lists, address="address", prefix="prefix"):
    offsets = net_lists.offsets.data.astype("uint64")
    cont = (
        net_lists.content.content if net_lists.content.is_option else net_lists.content
    )
    (arr,) = to_ip4(cont[address])
    out_addr, out_pref, counts = lib.aggregate4(arr, offsets, cont[prefix].data)
    # TODO: reassemble optional if input net_lists was list[optional[networks]]
    return contents.ListOffsetArray(
        index.Index64(counts),
        contents.RecordArray(
            [utils.u8_to_ip4(out_addr), contents.NumpyArray(out_pref)],
            fields=[address, prefix],
        ),
    )


def to_int_list(arr):
    if arr.is_leaf and arr.dtype.itemsize == 4:
        out = contents.RegularArray(contents.NumpyArray(arr.data.view("uint8")), size=4)
    else:
        out = ak.copy(arr)
        out.parameters.pop("__array__")
    return out


def to_bytestring(arr):
    if arr.is_leaf and arr.dtype.itemsize == 4:
        out = utils.u8_to_ip4(arr)
    else:
        out = ak.copy(arr)
        out.parameters["__array__"] = "bytestring"
        out.content.parameters["__array__"] = "byte"
    return out


def to_ip4(arr):
    if arr.is_leaf:
        # any 4-byte type like uint32
        return (arr.data.view("uint32"),)
    else:
        # bytestring or 4 * uint8 regular
        return (arr.content.data.view("uint32"),)


def to_ip6(arr):
    # always pass as bytes, and assume length is mod 16 in rust
    return (arr.content.data.view("uint8"),)


def dec_ip(func, conv=to_ip4, match=match_ip4, outtype=contents.NumpyArray):
    @functools.wraps(func)
    def func1(arr):
        return func(*conv(arr))

    return dec(func1, match=match, outtype=outtype, inmode="ak")


def bitwise_or(arr, other):
    if isinstance(other, (str, int)):
        other = ak.Array(
            np.array(list(ipaddress.ip_address("255.0.0.0").packed), dtype="uint8")
        )
    out = (ak.without_parameters(arr) | ak.without_parameters(other)).layout
    out.parameters["__array__"] = "bytestring"
    out.content.parameters["__array__"] = "byte"
    return out


def bitwise_and(arr, other):
    if isinstance(other, (str, int)):
        other = ak.Array(
            np.array(list(ipaddress.ip_address("255.0.0.0").packed), dtype="uint8")
        )
    out = (ak.without_parameters(arr) | ak.without_parameters(other)).layout
    out.parameters["__array__"] = "bytestring"
    out.content.parameters["__array__"] = "byte"
    return out


def equals(arr, other):
    if isinstance(other, (str, int)):
        other = ak.Array([ipaddress.ip_address(other).packed])

    return arr == other


class IPAccessor:
    def __eq__(self, *_):
        return dec(equals, match=match_ip, inmode="ak")

    bitwise_or = staticmethod(dec(bitwise_or, inmode="ak", match=match_ip))

    def __or__(self, *_):
        return dec(bitwise_or, match=match_ip, inmode="ak")

    def __ror__(self, *_):
        return dec(bitwise_or, match=match_ip, inmode="ak")

    bitwise_and = staticmethod(dec(bitwise_and, inmode="ak", match=match_ip))

    def __and__(self, *_):
        return dec(bitwise_and, match=match_ip, inmode="ak")

    def __rand__(self, *_):
        return dec(bitwise_and, match=match_ip, inmode="ak")

    to_int_list = staticmethod(dec(to_int_list, inmode="ak", match=match_ip))
    to_bytestring = staticmethod(dec(to_bytestring, inmode="ak", match=match_ip))

    is_unspecified4 = staticmethod(dec_ip(lib.is_unspecified4))
    is_broadcast4 = staticmethod(dec_ip(lib.is_broadcast4))
    is_global4 = staticmethod(dec_ip(lib.is_global4))
    is_loopback4 = staticmethod(dec_ip(lib.is_loopback4))
    is_private4 = staticmethod(dec_ip(lib.is_private4))
    is_link_local4 = staticmethod(dec_ip(lib.is_link_local4))
    is_shared4 = staticmethod(dec_ip(lib.is_shared4))
    is_benchmarking4 = staticmethod(dec_ip(lib.is_benchmarking4))
    is_reserved4 = staticmethod(dec_ip(lib.is_reserved4))
    is_multicast4 = staticmethod(dec_ip(lib.is_multicast4))
    is_documentation4 = staticmethod(dec_ip(lib.is_documentation4))

    to_string4 = staticmethod(dec_ip(lib.to_text4, outtype=utils.to_ak_string))

    parse_address4 = staticmethod(
        dec(parse_address4, inmode="ak", match=match_stringlike)
    )
    parse_net4 = staticmethod(dec(parse_net4, inmode="ak", match=match_stringlike))
    network4 = staticmethod(dec(network4, inmode="ak", match=match_net4))
    hostmask4 = staticmethod(dec(hostmask4, inmode="ak", match=match_net4))
    netmask4 = staticmethod(dec(netmask4, inmode="ak", match=match_net4))
    broadcast4 = staticmethod(dec(broadcast4, inmode="ak", match=match_net4))
    trunc4 = staticmethod(dec(trunc4, inmode="ak", match=match_net4))
    supernet4 = staticmethod(dec(supernet4, inmode="ak", match=match_net4))
    subnets4 = staticmethod(dec(subnets4, inmode="ak", match=match_net4))
    aggregate4 = staticmethod(dec(aggregate4, inmode="ak", match=match_list_net4))

    contains4 = staticmethod(dec(contains4, inmode="ak", match=match_net4))

    to_ipv6_mapped = staticmethod(dec_ip(lib.to_ipv6_mapped, outtype=utils.u8_to_ip6))

    hosts4 = staticmethod(dec(hosts4, match=match_net4, inmode="ak"))

    is_benchmarking6 = staticmethod(
        dec_ip(lib.is_benchmarking6, conv=to_ip6, match=match_ip6)
    )
    is_global6 = staticmethod(dec_ip(lib.is_global6, conv=to_ip6, match=match_ip6))
    is_documentation6 = staticmethod(
        dec_ip(lib.is_documentation6, conv=to_ip6, match=match_ip6)
    )
    is_unspecified6 = staticmethod(
        dec_ip(lib.is_unspecified6, conv=to_ip6, match=match_ip6)
    )
    is_loopback6 = staticmethod(dec_ip(lib.is_loopback6, conv=to_ip6, match=match_ip6))
    is_multicast6 = staticmethod(
        dec_ip(lib.is_multicast6, conv=to_ip6, match=match_ip6)
    )
    is_unicast6 = staticmethod(dec_ip(lib.is_unicast6, conv=to_ip6, match=match_ip6))
    is_ipv4_mapped = staticmethod(
        dec_ip(lib.is_ipv4_mapped, conv=to_ip6, match=match_ip6)
    )
    is_unicast_link_local = staticmethod(
        dec_ip(lib.is_unicast_link_local, conv=to_ip6, match=match_ip6)
    )
    is_unique_local = staticmethod(
        dec_ip(lib.is_unique_local, conv=to_ip6, match=match_ip6)
    )

    to_string6 = staticmethod(
        dec_ip(lib.to_text6, conv=to_ip6, match=match_ip6, outtype=utils.to_ak_string)
    )
    parse_address6 = staticmethod(
        dec(parse_address6, inmode="ak", match=match_stringlike)
    )


EagerAccessor.register_accessor("ip", IPAccessor)
LazyAccessor.register_accessor("ip", IPAccessor)
