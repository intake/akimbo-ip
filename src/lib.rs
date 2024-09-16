#![feature(ip)]
#![feature(addr_parse_ascii)]
use pyo3::prelude::*;
use core::net::Ipv4Addr;
use std::net::Ipv6Addr;
use std::str::{self, FromStr};
use ipnet::Ipv4Net;
use numpy::pyo3::Python;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};


pub fn netmask_to_prefix4(mask: u32) -> u8 {
    mask.leading_ones() as u8
}

pub fn netmask_to_prefix6(mask: u128) -> u8 {
    mask.leading_ones() as u8
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

#[pyfunction]
fn to_text6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) 
-> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u32>>)> {
    let mut offsets: Vec<u32> = vec!(0, );
    let mut data: Vec<u8> = Vec::new();
    for sl in  x.as_slice().unwrap().chunks_exact(16)
        {
            data.extend(Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).to_string().as_bytes());
            offsets.push(data.len() as u32);
        };
    Ok((data.into_pyarray_bound(py), offsets.into_pyarray_bound(py)))
}

#[pyfunction]
fn parse6<'py>(py: Python<'py>, offsets: PyReadonlyArray1<'py, u32>,
            data : PyReadonlyArray1<'py, u8>
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let ar = offsets.as_array();
    let sl = ar.as_slice().unwrap();
    let ar2 = data.as_array();
    let by = ar2.as_slice().unwrap();
    let mut out: Vec<u8> = Vec::with_capacity((sl.len() - 1) * 16);
    for w in  sl.windows(2) {
        out.extend(Ipv6Addr::parse_ascii(&by[w[0] as usize..w[1] as usize]).unwrap().octets())
    };
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


// list of IP4 addresses indicated by each network
#[pyfunction]
fn hosts4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
    let mut out: Vec<u32> = Vec::new();
    let mut offsets: Vec<u64> = Vec::from([0]);
    for (&add, &pre) in addr.as_array().iter().zip(pref.as_array()) {
        let hosts = Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().hosts();
        out.extend(hosts.map(|ip|ip.to_bits()));
        offsets.push(out.len() as u64);
    };
    Ok((out.into_pyarray_bound(py), offsets.into_pyarray_bound(py)))
}

/// the hostmask implied by the given network prefix
#[pyfunction]
fn hostmask4<'py>(py: Python<'py>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = pref.as_array().iter().map(
        |x| u32::max_value() >> x
    ).collect();
    Ok(out.into_pyarray_bound(py))
}


/// the netmask implied by the given network prefix
#[pyfunction]
fn netmask4<'py>(py: Python<'py>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    // TODO: check for prefix >= 128 .checked_shl(prefix).unwrap_or(0)
    let out: Vec<u32> = pref.as_array().iter().map(
        |x| u32::max_value() << (32 - x)
    ).collect();
    Ok(out.into_pyarray_bound(py))
}

/// the base network address of the given network values
#[pyfunction]
fn network4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr.as_array().iter().zip(pref.as_array().iter()).map(
        | (&add, &pre) | Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().network().to_bits()
    ).collect();
    Ok(out.into_pyarray_bound(py))
}


/// the highest address of the given network values
#[pyfunction]
fn broadcast4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr.as_array().iter().zip(pref.as_array().iter()).map(
        | (&add, &pre) | Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().broadcast().to_bits()
    ).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn trunc4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr.as_array().iter().zip(pref.as_array().iter()).map(
        | (&add, &pre) | Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().trunc().addr().to_bits()
    ).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn supernet4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr.as_array().iter().zip(pref.as_array().iter()).map(
        | (&add, &pre) | Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().supernet().unwrap().addr().to_bits()
    ).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn subnets4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
    new_pref: u8
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
    let mut out: Vec<u32> = Vec::new();
    let mut counts: Vec<u64> = Vec::with_capacity(pref.len());
    let mut count: u64 = 0;
    counts.push(0);
    addr.as_array().iter().zip(pref.as_array().iter()).for_each(
        | (&add, &pre) | {
            Ipv4Net::new(Ipv4Addr::from_bits(add), pre).unwrap().subnets(new_pref).unwrap().for_each(
                |x|{
                    count += 1;
                    out.push(x.addr().to_bits())
                }
            );
            counts.push(count);
        }
        
    );
    Ok((out.into_pyarray_bound(py), counts.into_pyarray_bound(py)))
}

#[pyfunction]
fn aggregate4<'py>(py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    offsets: PyReadonlyArray1<'py, u64>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u64>>)> {
    let mut out_addr: Vec<u32> = Vec::new();
    let mut out_pref: Vec<u8> = Vec::new();
    let mut counts: Vec<u64> = Vec::with_capacity(pref.len());
    let mut count: u64 = 0;
    let mut count_in: u64 = 0;
    let mut networks: Vec<Ipv4Net> = Vec::new();

    let off_arr = offsets.as_array();
    let offs = off_arr.as_slice().unwrap();
    let ad_arr = addr.as_array();
    let mut ad_slice = ad_arr.as_slice().unwrap().iter();
    let pr_arr = pref.as_array();
    let mut pr_slice = pr_arr.as_slice().unwrap().iter();

    for w in offs {
        networks.clear();
        while count_in < *w {
            networks.push(Ipv4Net::new(Ipv4Addr::from_bits(*ad_slice.next().unwrap()), *pr_slice.next().unwrap()).unwrap());
            count_in += 1;
        };
        Ipv4Net::aggregate(&networks).iter().for_each(
            |x| {
                out_addr.push(x.addr().to_bits());
                out_pref.push(x.prefix_len());
                count += 1;
            });
        counts.push(count);
    }
    Ok((out_addr.into_pyarray_bound(py), out_pref.into_pyarray_bound(py), counts.into_pyarray_bound(py)))
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
fn is_benchmarking6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_benchmarking()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_documentation6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_documentation()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_global6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_global()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_ipv4_mapped<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_ipv4_mapped()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_loopback6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_loopback()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_multicast6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_multicast()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_unicast6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unicast()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_unicast_link_local<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unicast_link_local()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_unique_local<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unique_local()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn is_unspecified6<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u8>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x.as_slice().unwrap().chunks_exact(16).map(|sl | {
        Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unspecified()
    }).collect();
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn to_ipv6_mapped<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, u32>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let mut out: Vec<u8> = Vec::with_capacity(x.len() * 16);
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
    m.add_function(wrap_pyfunction!(hostmask4, m)?)?;
    m.add_function(wrap_pyfunction!(netmask4, m)?)?;
    m.add_function(wrap_pyfunction!(network4, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast4, m)?)?;
    m.add_function(wrap_pyfunction!(trunc4, m)?)?;
    m.add_function(wrap_pyfunction!(supernet4, m)?)?;
    m.add_function(wrap_pyfunction!(subnets4, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate4, m)?)?;

    m.add_function(wrap_pyfunction!(is_benchmarking6, m)?)?;
    m.add_function(wrap_pyfunction!(is_documentation6, m)?)?;
    m.add_function(wrap_pyfunction!(is_global6, m)?)?;
    m.add_function(wrap_pyfunction!(is_ipv4_mapped, m)?)?;
    m.add_function(wrap_pyfunction!(is_loopback6, m)?)?;
    m.add_function(wrap_pyfunction!(is_multicast6, m)?)?;
    m.add_function(wrap_pyfunction!(is_unicast6, m)?)?;
    m.add_function(wrap_pyfunction!(is_unicast_link_local, m)?)?;
    m.add_function(wrap_pyfunction!(is_unique_local, m)?)?;
    m.add_function(wrap_pyfunction!(is_unspecified6, m)?)?;
    m.add_function(wrap_pyfunction!(to_text6, m)?)?;
    m.add_function(wrap_pyfunction!(parse6, m)?)?;
    Ok(())
}