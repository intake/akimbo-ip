#![feature(ip)]
#![feature(addr_parse_ascii)]
use pyo3::prelude::*;
use core::net::Ipv4Addr;
use std::str::{self, FromStr};
use ipnet::Ipv4Net;
use numpy::pyo3::Python;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};


pub fn netmask_to_prefix4(mask: u32) -> u8 {
    mask.leading_ones() as u8
}

pub fn prefix_to_netmask4(prefix: u8) -> u32 {
    // TODO: check for prefix >= 32 .checked_shl(prefix).unwrap_or(0)
    0xffffffff << prefix
}

pub fn netmask_to_prefix6(mask: u128) -> u8 {
    mask.leading_ones() as u8
}

pub fn prefix_to_netmask6(prefix: u8) -> u128 {
    // TODO: check for prefix >= 128 .checked_shl(prefix).unwrap_or(0)
    0xffffffffffffffffffffffffffffffff << prefix
}


#[pyfunction]
fn to_text4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) 
-> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u32>>)> {
    let mut offsets: Vec<u32> = vec!(0, );
    let mut data: Vec<u8> = Vec::new();
    for out in  x.as_array().iter()
        {
            data.extend(Ipv4Addr::from_bits(*out).to_string().as_bytes());
            offsets.push(data.len() as u32);
        };
    Ok((data.into_pyarray_bound(py), offsets.into_pyarray_bound(py)))
}

/// Parse strings into IP4 addresses (length 4 bytestrings)
#[pyfunction]
fn parse4<'py>(py: Python<'py>, offsets: PyReadonlyArray1<'py, u32>,
            data : PyReadonlyArray1<'py, u8>
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let ar = offsets.as_array();
    let sl = ar.as_slice().unwrap();
    let ar2 = data.as_array();
    let by = ar2.as_slice().unwrap();
    let out: Vec<u32> = sl.windows(2).map(
        |w| {
            Ipv4Addr::parse_ascii(&by[w[0] as usize..w[1] as usize]).unwrap().to_bits()
        }
    ).collect();
    Ok(out.into_pyarray_bound(py))
}

/// Parse strings into IP4 networks (length 4 bytestring and 1-byte prefix value)
#[pyfunction]
fn parsenet4<'py>(py: Python<'py>, 
    offsets: PyReadonlyArray1<'py, u32>,
    data : PyReadonlyArray1<'py, u8>
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u8>>)> {
    let ar = offsets.as_array();
    let sl = ar.as_slice().unwrap();
    let ar2 = data.as_array();
    let by = ar2.as_slice().unwrap();
    let mut outaddr: Vec<u32> = Vec::with_capacity(ar.len() - 1);
    let mut outpref: Vec<u8> = Vec::with_capacity(ar.len() - 1);
    for w in sl.windows(2) {
        let net = Ipv4Net::from_str(
            &str::from_utf8(&by[w[0] as usize..w[1] as usize]).unwrap()).unwrap();
        outaddr.push(net.addr().to_bits());
        outpref.push(net.prefix_len());
    };
    Ok((outaddr.into_pyarray_bound(py), outpref.into_pyarray_bound(py)))
}


/// Is `other` contained in the address/prefix pairs of the input array?
#[pyfunction]
fn contains_one4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
    other: u32
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = addr.as_array().iter().zip(pref.as_array()).map(|(add, pre)| 
        Ipv4Net::new(Ipv4Addr::from_bits(*add), *pre).unwrap().contains(&Ipv4Addr::from_bits(other))
    ).collect();
    Ok(out.into_pyarray_bound(py))
}


#[pyfunction]
fn hosts4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
// returns IP4 data as uint32 and array of offsets (same length as input)
    let mut out: Vec<u32> = Vec::new();
    let mut offsets: Vec<u64> = Vec::from([0]);
    for (&add, &pre) in addr.as_array().iter().zip(pref.as_array()) {
        let hosts = Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().hosts();
        out.extend(hosts.map(|ip|ip.to_bits()));
        offsets.push(out.len() as u64);
    };
    Ok((out.into_pyarray_bound(py), offsets.into_pyarray_bound(py)))
}


#[pyfunction]
fn is_broadcast4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_broadcast()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_global4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_global()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_unspecified4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_unspecified()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_loopback4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_loopback()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_private4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_private()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_link_local4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_link_local()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_shared4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_shared()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_benchmarking4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_benchmarking()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_reserved4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_reserved()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_multicast4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_multicast()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_documentation4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_array().iter().map(|&x|Ipv4Addr::from_bits(x).is_documentation()).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn to_ipv6_mapped<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let mut out: Vec<u8> = Vec::with_capacity(x.len().unwrap() * 16);
    for &x in x.as_array().iter() {
        let bit = Ipv4Addr::from(x).to_ipv6_mapped().octets();
        out.extend(bit);
    };
    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn akimbo_ip(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_broadcast4, m)?)?;
    m.add_function(wrap_pyfunction!(is_unspecified4, m)?)?;
    m.add_function(wrap_pyfunction!(is_global4, m)?)?;
    m.add_function(wrap_pyfunction!(is_loopback4, m)?)?;
    m.add_function(wrap_pyfunction!(is_private4, m)?)?;
    m.add_function(wrap_pyfunction!(is_link_local4, m)?)?;
    m.add_function(wrap_pyfunction!(is_shared4, m)?)?;
    m.add_function(wrap_pyfunction!(is_benchmarking4, m)?)?;
    m.add_function(wrap_pyfunction!(is_reserved4, m)?)?;
    m.add_function(wrap_pyfunction!(is_multicast4, m)?)?;
    m.add_function(wrap_pyfunction!(is_documentation4, m)?)?;
    m.add_function(wrap_pyfunction!(to_text4, m)?)?;
    m.add_function(wrap_pyfunction!(parse4, m)?)?;
    m.add_function(wrap_pyfunction!(parsenet4, m)?)?;
    m.add_function(wrap_pyfunction!(contains_one4, m)?)?;
    m.add_function(wrap_pyfunction!(to_ipv6_mapped, m)?)?;
    m.add_function(wrap_pyfunction!(hosts4, m)?)?;
    Ok(())
}