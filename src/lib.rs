#![feature(ip)]
#![feature(addr_parse_ascii)]
use pyo3::prelude::*;
use core::net::Ipv4Addr;
use std::str::{self, FromStr};
use ipnet::Ipv4Net;
use numpy::pyo3::Python;
use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};


pub fn netmask_to_prefix4(mut mask: u32) -> u8 {
    let mut zerobits: u8 = 0;
    while (mask & 0x1) == 0 {
        mask = mask >> 1;
        zerobits += 1;
    }
    32 - zerobits
}

pub fn prefix_to_netmask4(prefix: u8) -> u32 {
    // TODO: check for prefix >= 32 .checked_shl(prefix).unwrap_or(0)
    0xffffffff << prefix
}


pub fn netmask_to_prefix6(mut mask: u128) -> u8 {
    let mut zerobits: u8 = 0;
    while (mask & 0x1) == 0 {
        mask = mask >> 1;
        zerobits += 1;
    }
    32 - zerobits
}

pub fn prefix_to_netmask6(prefix: u8) -> u128 {
    // TODO: check for prefix >= 128 .checked_shl(prefix).unwrap_or(0)
    0xffffffffffffffffffffffffffffffff << prefix
}


/// Simplistic function that prints out the input ints as IP4 addresses
#[pyfunction]
fn print_out<'py>(x: PyReadonlyArray1<'py, u32>) -> PyResult<()> {
    let x: ArrayView1<u32> = x.as_array();
    for val in x.iter() {
        let ip = Ipv4Addr::from(*val);
        println!("{:?}", ip)
    }
    Ok(())
}

#[pyfunction]
fn to_text4<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) 
-> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u32>>)> {
    let mut offsets: Vec<u32> = vec!(0, );
    let mut data: Vec<u8> = Vec::new();
    let mut curr: u32 = 0;
    for out in  x.as_array().iter()
        {
            let s = Ipv4Addr::from_bits(*out).to_string();
            let t = s.as_bytes();
            data.extend(t);
            curr += t.len() as u32;
            offsets.push(curr);
        };
    Ok((data.into_pyarray_bound(py), offsets.into_pyarray_bound(py)))
}

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

//#[pyfunction]
//fn to_ipv6_compatible<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<u128>>> {
//    let out: Vec<u128> = x.as_array().iter().map(|&x|Ipv4Addr::from(x).to_ipv6_compatible()).collect();
//    Ok(out.into_pyarray_bound(py))
//}

#[pymodule]
fn akimbo_ip(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_out, m)?)?;
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
    Ok(())
}