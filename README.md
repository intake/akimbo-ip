Akimbo-ip
=========

Extension enabling fast vector processing of IPv4 and IPv6 values
within nested/ragged columns of dataframes.

(experimental)

Installation
------------

Run one of the following

```bash
> pip install git+https://github.com/intake/akimbo-ip  # dev version
> pip install akimbo-ip  # released version
```

Model
-----

- IPv4 addresses are (fixed) length 4 bytestrings, but can be represented
  by any 4-bye value, e.g., uint32 or fixed-4-length list of uint8
- IPv6 addresses are (fixed) length 16 bytestrings or fixed-16-length list
  of uint8
- Networks are records with an IPv4 or IPv6 field (nominally "address") and
  a uint8 field for the prefix length (nominally "prefix"). The field
  names can be overidden.

We can convert between hostmasks, netmasks and prefix lengths. Some methods
require composite types like list-of-addresses, see the individual docstrings.

As with the normal functioning of akimbo, you can indicate which parts of
a nested structure should be with the `where=` kwargs to any method.

Usage
-----

```python
>>> import akimbo.pandas
>>> import akimbo_ip
```

This will anable the ``.ak`` accessor for ``pandas`` series and dataframes
(or pick a different dataframe library) and a subaccessor ``.ak.ip`` which
makes available several methods that act on IP addresses and network.
