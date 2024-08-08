Akimbo-ip
---------

Extension enabling fast vector processing of IPv4 and IPv6 values
within nested/ragged columns of dataframes.

(experimental)

Installation
~~~~~~~~~~~~

```bash
> pip install git+https://github.com/intake/akimbo-ip
```

Model
~~~~~

- IPv4 addresses are (fixed) length 4 bytestrings, but can be represented
  by any 4-bye value, e.g., uint32
- IPv6 addresses are (fixed) length 16 bytestrings
- Networks are records with an IPv4 or IPv6 field (nominally "address") and
  a uint8 field for the prefix length (nominally "prefix")

We can convert between hostmasks, netmasks and prefix lengths.

Some methods 

Usage
~~~~~

```python
>>> import akimbo.pandas
>>> import akimbo_ip
```

This will anable the ``.ak`` accessor for ``pandas`` series and dataframes
(or pick a different dataframe library) and a subaccessor ``.ak.ip`` which
makes available several methods that act on IP addresses and network.
