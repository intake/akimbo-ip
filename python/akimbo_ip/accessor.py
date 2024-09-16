import ipaddress
import functools
from types import UnionType

import awkward as ak
import numpy as np
import pyarrow as pa

from akimbo.mixin import Accessor
from akimbo.apply_tree import dec
import akimbo_ip.akimbo_ip as lib
from akimbo_ip import utils


def match_ip4(arr):
    """matches fixed-list[4, u8] and fixed-bytestring[4] and ANY 4-byte value (like uint32, assumed big-endian"""
    return (arr.is_leaf and arr.dtype.itemsize == 4) or (
        arr.is_regular and arr.size == 4 and arr.content.is_leaf and arr.content.dtype.itemsize == 1)


def match_ip6(arr):
    """matches fixed-list[16, u8] and fixed-bytestring[16]"""
    return arr.is_regular and arr.size == 16 and arr.content.is_leaf and arr.content.dtype.itemsize == 1


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
    out = lib.parse4(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return utils.u8_to_ip4(out.view("uint8"))


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
    out = lib.parsenet4(
        str_arr.offsets.data.astype("uint32"), str_arr.content.data
    )
    return ak.contents.RecordArray(
        [ak.contents.RegularArray(
            ak.contents.NumpyArray(out[0].view("uint8"), parameters={"__array__": "byte"}), 
            size=4, 
            parameters={"__array__": "bytestring"}
        ),
        ak.contents.NumpyArray(out[1])],
        fields=["address", "prefix"]
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
    return ak.contents.NumpyArray(out)


def hosts4(nets, address="address", prefix="prefix"):
    arr, = to_ip4(nets[address])
    ips, offsets = lib.hosts4(arr, nets[prefix].data.astype("uint8"))
    return ak.contents.ListOffsetArray(
        ak.index.Index64(offsets),
        utils.u8_to_ip4(ips)
    )

def network4(nets, address="address", prefix="prefix"):
    arr, = to_ip4(nets[address])
    out = lib.network4(arr, nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)
    

def broadcast4(nets, address="address", prefix="prefix"):
    arr, = to_ip4(nets[address])
    out = lib.broadcast4(arr, nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)


def hostmask4(nets, address="address", prefix="prefix"):
    out = lib.hostmask4(nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)
    

def netmask4(nets, address="address", prefix="prefix"):
    out = lib.netmask4(nets[prefix].data.astype("uint8"))
    return utils.u8_to_ip4(out)


def trunc4(nets, address="address", prefix="prefix"):
    arr, = to_ip4(nets[address])
    out = lib.trunc4(arr, nets[prefix].data.astype("uint8"))
    return ak.contents.RecordArray(
        [utils.u8_to_ip4(out), nets[prefix]],
        fields=[address, prefix]
    )
    

def supernet4(nets, address="address", prefix="prefix"):
    arr, = to_ip4(nets[address])
    out = lib.supernet4(arr, nets[prefix].data.astype("uint8"))
    return ak.contents.RecordArray(
        [utils.u8_to_ip4(out), ak.contents.NumpyArray(nets[prefix].data - 1)],
        fields=[address, prefix]
    )
    

def subnets4(nets, new_prefix, address="address", prefix="prefix"):
    arr, = to_ip4(nets[address])
    out, offsets = lib.subnets4(arr, nets[prefix].data.astype("uint8"), new_prefix)
    addr = utils.u8_to_ip4(out)
    return ak.contents.ListOffsetArray(
        ak.index.Index64(offsets),
        ak.contents.RecordArray(
            [addr, 
             ak.contents.NumpyArray(np.full((len(addr), ), new_prefix, dtype="uint8"))],
            fields=[address, prefix]
        ),
    )
 

def aggregate4(net_lists, address="address", prefix="prefix"):
    offsets = net_lists.offsets.data.astype("uint64")
    cont = net_lists.content.content if net_lists.content.is_option else net_lists.content
    arr, = to_ip4(cont[address])
    out_addr, out_pref, counts = lib.aggregate4(arr, offsets, cont[prefix].data)
    # TODO: reassemble optional if input net_lists was list[optional[networks]]
    return ak.contents.ListOffsetArray(
        ak.index.Index64(counts),
        ak.contents.RecordArray(
            [utils.u8_to_ip4(out_addr), ak.contents.NumpyArray(out_pref)],
            fields=[address, prefix]
        )
    )
    

def to_int_list(arr):
    if (arr.is_leaf and arr.dtype.itemsize == 4):
        out = ak.contents.RegularArray(
            ak.contents.NumpyArray(arr.data.view('uint8')),
            size=4
        )
    else:
        out = ak.copy(arr)
        out.parameters.pop('__array__')
    return out


def to_bytestring(arr):
    if (arr.is_leaf and arr.dtype.itemsize == 4):
        out = utils.u8_to_ip4(arr)
    else:
        out = ak.copy(arr)
        out.parameters['__array__'] = "bytestring"
        out.content.parameters["__array__"] = "byte"
    return out


def to_ip4(arr):
    if arr.is_leaf:
        return arr.data.view("uint32"),
    else:
        # bytestring or 4 * uint8 regular
        return arr.content.data.view("uint32"),

def to_ip6(arr):
    # always pass as bytes, and assume length is mod 16 in rust
    return arr.content.data.view("uint8"),
    

def dec_ip(func, conv=to_ip4, match=match_ip4, outtype=ak.contents.NumpyArray):
    @functools.wraps(func)
    def func1(arr):
        return func(*conv(arr))

    return dec(func1, match=match, outtype=outtype, inmode="awkward")


def bitwise_or(arr, other):
    if isinstance(other, (str, int)):
        other = ak.Array(np.array(list(ipaddress.ip_address("255.0.0.0").packed), dtype="uint8"))
    out = (ak.without_parameters(arr) | ak.without_parameters(other)).layout
    out.parameters["__array__"] = "bytestring"
    out.content.parameters["__array__"] = "byte"
    return out
            

def bitwise_and(arr, other):
    if isinstance(other, (str, int)):
        other = ak.Array(np.array(list(ipaddress.ip_address("255.0.0.0").packed), dtype="uint8"))
    out = (ak.without_parameters(arr) | ak.without_parameters(other)).layout
    out.parameters["__array__"] = "bytestring"
    out.content.parameters["__array__"] = "byte"
    return out


class IPAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    # TODO: bitwise_or and bitwise_and methods and their overrides
    def __eq__(self, other):
        arr = self.accessor.array
        if isinstance(other, (str, int)):
            arr2 = ak.Array([ipaddress.ip_address(other).packed])
            
            return self.accessor.to_output(arr == arr2)
        else:
            raise ValueError

    bitwise_or = dec(bitwise_or, inmode="ak", match=match_ip)
    
    __or__ = bitwise_or
    def __ror__(self, value):
        return self.__or__(value)

    bitwise_and = dec(bitwise_and, inmode="ak", match=match_ip)
    
    __and__ = bitwise_and
    def __rand__(self, value):
        return self.__and__(value)

    to_int_list = dec(to_int_list, inmode="ak", match=match_ip)
    to_bytestring = dec(to_bytestring, inmode="ak", match=match_ip)

    is_unspecified4 = dec_ip(lib.is_unspecified4)
    is_broadcast4 = dec_ip(lib.is_broadcast4)
    is_global4 = dec_ip(lib.is_global4)
    is_loopback4 = dec_ip(lib.is_loopback4)
    is_private4 = dec_ip(lib.is_private4)
    is_link_local4 = dec_ip(lib.is_link_local4)
    is_shared4 = dec_ip(lib.is_shared4)
    is_benchmarking4 = dec_ip(lib.is_benchmarking4)
    is_reserved4 = dec_ip(lib.is_reserved4)
    is_multicast4 = dec_ip(lib.is_multicast4)
    is_documentation4 = dec_ip(lib.is_documentation4)

    to_string4 = dec_ip(lib.to_text4, outtype=utils.to_ak_string)

    parse_address4 = dec(parse_address4, inmode="ak", match=match_stringlike)
    parse_net4 = dec(parse_net4, inmode="ak", match=match_stringlike)
    network4 = dec(network4, inmode="ak", match=match_net4)
    hostmask4 = dec(hostmask4, inmode="ak", match=match_net4)
    netmask4 = dec(netmask4, inmode="ak", match=match_net4)
    broadcast4 = dec(broadcast4, inmode="ak", match=match_net4)
    trunc4 = dec(trunc4, inmode="ak", match=match_net4)
    supernet4 = dec(supernet4, inmode="ak", match=match_net4)
    subnets4 = dec(subnets4, inmode="ak", match=match_net4)
    aggregate4 = dec(aggregate4, inmode="ak", match=match_list_net4)
    
    contains4 = dec(contains4, inmode="ak", match=match_net4)

    to_ipv6_mapped = dec_ip(lib.to_ipv6_mapped, outtype=utils.u8_to_ip6)

    hosts4 = dec(hosts4, match=match_net4, inmode="ak")

    is_benchmarking6 = dec_ip(lib.is_benchmarking6, conv=to_ip6, match=match_ip6)
    is_global6 = dec_ip(lib.is_global6, conv=to_ip6, match=match_ip6)
    is_documentation6 = dec_ip(lib.is_documentation6, conv=to_ip6, match=match_ip6)
    is_unspecified6 = dec_ip(lib.is_unspecified6, conv=to_ip6, match=match_ip6)
    is_loopback6 = dec_ip(lib.is_loopback6, conv=to_ip6, match=match_ip6)
    is_multicast6 = dec_ip(lib.is_multicast6, conv=to_ip6, match=match_ip6)
    is_unicast6 = dec_ip(lib.is_unicast6, conv=to_ip6, match=match_ip6)
    is_ipv4_mapped = dec_ip(lib.is_ipv4_mapped, conv=to_ip6, match=match_ip6)
    is_unicast_link_local = dec_ip(lib.is_unicast_link_local, conv=to_ip6, match=match_ip6)
    is_unique_local = dec_ip(lib.is_unique_local, conv=to_ip6, match=match_ip6)

    to_string6 = dec_ip(lib.to_text6, conv=to_ip6, match=match_ip6, outtype=utils.to_ak_string)
    parse_address6 = dec(parse_address6, inmode="ak", match=match_stringlike)


Accessor.register_accessor("ip", IPAccessor)
