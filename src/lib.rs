#![feature(ip)]
#![feature(addr_parse_ascii)]
use core::net::Ipv4Addr;
use ipnet::Ipv4Net;
use numpy::pyo3::Python;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::net::Ipv6Addr;
use std::str::{self, FromStr};

pub fn netmask_to_prefix4(mask: u32) -> u8 {
    mask.leading_ones() as u8
}

pub fn netmask_to_prefix6(mask: u128) -> u8 {
    mask.leading_ones() as u8
}

#[pyfunction]
fn to_text4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u32>>)> {
    let mut offsets: Vec<u32> = vec![0];
    let mut data: Vec<u8> = Vec::new();
    for out in x.as_array().iter() {
        let (a, b, c, d) = out.to_be_bytes().into();
        data.extend(Ipv4Addr::new(a, b, c, d).to_string().as_bytes());
        offsets.push(data.len() as u32);
    }
    Ok((data.into_pyarray(py), offsets.into_pyarray(py)))
}

/// Parse strings into IP4 addresses (length 4 bytestrings)
#[pyfunction]
fn parse4<'py>(
    py: Python<'py>,
    offsets: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u8>>)> {
    let ar = offsets.as_array();
    let sl = ar.as_slice().unwrap();
    let ar2 = data.as_array();
    let by = ar2.as_slice().unwrap();
    let (out, valid): (Vec<u32>, Vec<u8>) = sl
        .windows(2)
        .map(
            |w| match Ipv4Addr::parse_ascii(&by[w[0] as usize..w[1] as usize]) {
                Ok(x) => (u32::from_ne_bytes(x.octets()), 1u8),
                Err(_) => (0u32, 0u8),
            },
        )
        .unzip();
    Ok((out.into_pyarray(py), valid.into_pyarray(py)))
}

#[pyfunction]
fn to_text6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u32>>)> {
    let mut offsets: Vec<u32> = vec![0];
    let mut data: Vec<u8> = Vec::new();
    for sl in x.as_slice().unwrap().chunks_exact(16) {
        data.extend(
            Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap()))
                .to_string()
                .as_bytes(),
        );
        offsets.push(data.len() as u32);
    }
    Ok((data.into_pyarray(py), offsets.into_pyarray(py)))
}

#[pyfunction]
fn parse6<'py>(
    py: Python<'py>,
    offsets: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let ar = offsets.as_array();
    let sl = ar.as_slice().unwrap();
    let ar2 = data.as_array();
    let by = ar2.as_slice().unwrap();
    let mut out: Vec<u8> = Vec::with_capacity((sl.len() - 1) * 16);
    for w in sl.windows(2) {
        out.extend(
            Ipv6Addr::parse_ascii(&by[w[0] as usize..w[1] as usize])
                .unwrap()
                .octets(),
        )
    }
    Ok(out.into_pyarray(py))
}

/// Parse strings into IP4 networks (length 4 bytestring and 1-byte prefix value)
#[pyfunction]
fn parsenet4<'py>(
    py: Python<'py>,
    offsets: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u8>>)> {
    let ar = offsets.as_array();
    let sl = ar.as_slice().unwrap();
    let ar2 = data.as_array();
    let by = ar2.as_slice().unwrap();
    let mut outaddr: Vec<u32> = Vec::with_capacity(ar.len() - 1);
    let mut outpref: Vec<u8> = Vec::with_capacity(ar.len() - 1);
    for w in sl.windows(2) {
        let net =
            Ipv4Net::from_str(&str::from_utf8(&by[w[0] as usize..w[1] as usize]).unwrap()).unwrap();
        outaddr.push(u32::from_ne_bytes(net.addr().octets()));
        outpref.push(net.prefix_len());
    }
    Ok((outaddr.into_pyarray(py), outpref.into_pyarray(py)))
}

/// Is `other` contained in the address/prefix pairs of the input array?
#[pyfunction]
fn contains_one4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
    other: u32,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = addr
        .as_array()
        .iter()
        .zip(pref.as_array())
        .map(|(add, pre)| {
            Ipv4Net::new(Ipv4Addr::from_bits(*add), *pre)
                .unwrap()
                .contains(&Ipv4Addr::from_bits(other))
        })
        .collect();
    Ok(out.into_pyarray(py))
}

// list of IP4 addresses indicated by each network
#[pyfunction]
fn hosts4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
    let mut out: Vec<u32> = Vec::new();
    let mut offsets: Vec<u64> = Vec::from([0]);
    for (&add, &pre) in addr.as_array().iter().zip(pref.as_array()) {
        let hosts = Ipv4Net::new(
            {
                let (a, b, c, d) = add.to_ne_bytes().into();
                Ipv4Addr::new(a, b, c, d)
            },
            pre,
        )
        .unwrap()
        .hosts();
        out.extend(hosts.map(|ip| u32::from_ne_bytes(ip.octets())));
        offsets.push(out.len() as u64);
    }
    Ok((out.into_pyarray(py), offsets.into_pyarray(py)))
}

/// the hostmask implied by the given network prefix
#[pyfunction]
fn hostmask4<'py>(
    py: Python<'py>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = pref
        .as_array()
        .iter()
        .map(|x| {
            u32::from_ne_bytes(
                Ipv4Net::new(Ipv4Addr::new(0, 0, 0, 0), *x)
                    .unwrap()
                    .hostmask()
                    .octets(),
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

/// the netmask implied by the given network prefix
#[pyfunction]
fn netmask4<'py>(
    py: Python<'py>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = pref
        .as_array()
        .iter()
        .map(|x| {
            u32::from_ne_bytes(
                Ipv4Net::new(Ipv4Addr::new(0, 0, 0, 0), *x)
                    .unwrap()
                    .netmask()
                    .octets(),
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

/// the base network address of the given network values
#[pyfunction]
fn network4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr
        .as_array()
        .iter()
        .zip(pref.as_array().iter())
        .map(|(&add, &pre)| {
            u32::from_ne_bytes(
                Ipv4Net::new(
                    {
                        let (a, b, c, d) = add.to_ne_bytes().into();
                        Ipv4Addr::new(a, b, c, d)
                    },
                    pre,
                )
                .unwrap()
                .network()
                .octets(),
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

/// the highest address of the given network values
#[pyfunction]
fn broadcast4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr
        .as_array()
        .iter()
        .zip(pref.as_array().iter())
        .map(|(&add, &pre)| {
            u32::from_ne_bytes(
                Ipv4Net::new(
                    {
                        let (a, b, c, d) = add.to_ne_bytes().into();
                        Ipv4Addr::new(a, b, c, d)
                    },
                    pre,
                )
                .unwrap()
                .broadcast()
                .octets(),
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn trunc4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr
        .as_array()
        .iter()
        .zip(pref.as_array().iter())
        .map(|(&add, &pre)| {
            u32::from_ne_bytes(
                Ipv4Net::new(
                    {
                        let (a, b, c, d) = add.to_ne_bytes().into();
                        Ipv4Addr::new(a, b, c, d)
                    },
                    pre,
                )
                .unwrap()
                .trunc()
                .addr()
                .octets(),
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn supernet4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let out: Vec<u32> = addr
        .as_array()
        .iter()
        .zip(pref.as_array().iter())
        .map(|(&add, &pre)| {
            u32::from_ne_bytes(
                Ipv4Net::new(
                    {
                        let (a, b, c, d) = add.to_ne_bytes().into();
                        Ipv4Addr::new(a, b, c, d)
                    },
                    pre,
                )
                .unwrap()
                .supernet()
                .unwrap()
                .addr()
                .octets(),
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn subnets4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    pref: PyReadonlyArray1<'py, u8>,
    new_pref: u8,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
    let mut out: Vec<u32> = Vec::new();
    let mut counts: Vec<u64> = Vec::with_capacity(pref.len());
    let mut count: u64 = 0;
    counts.push(0);
    addr.as_array()
        .iter()
        .zip(pref.as_array().iter())
        .for_each(|(&add, &pre)| {
            Ipv4Net::new(
                {
                    let (a, b, c, d) = add.to_ne_bytes().into();
                    Ipv4Addr::new(a, b, c, d)
                },
                pre,
            )
            .unwrap()
            .subnets(new_pref)
            .unwrap()
            .for_each(|x| {
                count += 1;
                out.push(u32::from_ne_bytes(x.addr().octets()))
            });
            counts.push(count);
        });
    Ok((out.into_pyarray(py), counts.into_pyarray(py)))
}

#[pyfunction]
fn aggregate4<'py>(
    py: Python<'py>,
    addr: PyReadonlyArray1<'py, u32>,
    offsets: PyReadonlyArray1<'py, u64>,
    pref: PyReadonlyArray1<'py, u8>,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u64>>,
)> {
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
            let (a, b, c, d): (u8, u8, u8, u8) = ad_slice.next().unwrap().to_ne_bytes().into();
            networks
                .push(Ipv4Net::new(Ipv4Addr::new(a, b, c, d), *pr_slice.next().unwrap()).unwrap());
            count_in += 1;
        }
        Ipv4Net::aggregate(&networks).iter().for_each(|x| {
            out_addr.push(u32::from_ne_bytes(x.addr().octets()));
            out_pref.push(x.prefix_len());
            count += 1;
        });
        counts.push(count);
    }
    Ok((
        out_addr.into_pyarray(py),
        out_pref.into_pyarray(py),
        counts.into_pyarray(py),
    ))
}

#[pyfunction]
fn is_broadcast4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_broadcast()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_global4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_global()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_unspecified4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_unspecified()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_loopback4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_loopback()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_private4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_private()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_link_local4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_link_local()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_shared4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_shared()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_benchmarking4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_benchmarking()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_reserved4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_reserved()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_multicast4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_multicast()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_documentation4<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_array()
        .iter()
        .map(|&x| {
            let (a, b, c, d) = x.to_ne_bytes().into();
            Ipv4Addr::new(a, b, c, d).is_documentation()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_benchmarking6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| {
            Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_benchmarking()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_documentation6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| {
            Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_documentation()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_global6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_global())
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_ipv4_mapped<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_ipv4_mapped())
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_loopback6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_loopback())
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_multicast6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_multicast())
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_unicast6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unicast())
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_unicast_link_local<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| {
            Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unicast_link_local()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_unique_local<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| {
            Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unique_local()
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn is_unspecified6<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let out: Vec<bool> = x
        .as_slice()
        .unwrap()
        .chunks_exact(16)
        .map(|sl| Ipv6Addr::from_bits(u128::from_be_bytes(sl.try_into().unwrap())).is_unspecified())
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn to_ipv6_mapped<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let mut out: Vec<u8> = Vec::with_capacity(x.len() * 16);
    for &x in x.as_array().iter() {
        let bit = Ipv4Addr::from(x).to_ipv6_mapped().octets();
        out.extend(bit);
    }
    Ok(out.into_pyarray(py))
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
