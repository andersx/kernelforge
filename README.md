# kernelforge
Optimized kernels for ML

- Without using F2PY or Meson

# Installation

```bash
pip install -e .
pytest -v -s
```

## Intel compilers and MKL

GNU compilers will be used by default. If you want to use Intel compilers and MKL, you can set the environment variables:

```bash
source /opt/intel/oneapi/setvars.sh
```
In this case, MKL will be autodetected and used. If you additionally want to compile with Intel compilers, you can set the environment variables when running `pip install`:

```bash
CC=icx CXX=icpx FC=ifx make install
```
