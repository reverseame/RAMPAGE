# RAMPAGE - A Training and Comparing AGD Detectors Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Version: v1.0](https://img.shields.io/badge/Version-v1.0.0-green.svg)]()

RAMPAGE: (*fRAMework to comPAre dGa dEtectors*) is a framework aimed at training and comparing machine learning models for the detection of Domain Generation Algorithms (DGAs).

## Installation

RAMPAGE runs on Python 3.11. To use RAMPAGE, it is necessary to create a pip package and install it. For now, it has been decided not to upload it to PyPI.

To generate the package, execute the following command:

```bash
python3 setup.py sdist bdist_wheel
```

The package will be generated in the `dist/` directory. To install it, use pip to install the file with the `.whl` extension that has been created. E.g.:

```
pip3 install dist/RAMPAGE-1.0-py3-none-any.whl --break-system-packages
```

## Usage

In the `examples/` directory, there is a sample execution that includes a series of models. Below is an explanation of how to set them up.

```python
*TODO*
```

## License

Licensed under the [GNU GPLv3](LICENSE) license.

## How to cite

If you are using this software, please cite it as follows:
```
TBD
```

## Funding support

Part of this research was supported by the Spanish National Cybersecurity Institute (INCIBE) under *Proyectos Estratégicos de Ciberseguridad -- CIBERSEGURIDAD EINA UNIZAR* and by the Recovery, Transformation and Resilience Plan funds, financed by the European Union (Next Generation).

![INCIBE_logos](misc/img/INCIBE_logos.jpg)