# RAMPAGE - A Training and Comparing AGD Detectors Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Version: v1.0](https://img.shields.io/badge/Version-v1.0.0-green.svg)]()

RAMPAGE: (*fRAMework to comPAre aGd dEtectors*) is a framework aimed at training and comparing machine learning models for the detection of Algorithmically Generated Domains (AGDs).

## Installation

RAMPAGE runs on Python 3.11. To use RAMPAGE, it is necessary to create a pip package and install it. For now, it has been decided not to upload it to PyPI.

To generate the package, execute the following command:

```bash
python3 setup.py sdist bdist_wheel
```

The package will be generated in the `dist/` directory. To install it, use pip to install the file with the `.whl` extension that has been created. E.g.:

```
pip3 install dist/RAMPAGE-1.0-py3-none-any.whl
```

## Usage

In the `examples/` directory, there is a sample execution that includes a series of models. Below is an explanation of how to set them up.

### Requirements

It is necessary to install the packages of the machine learning frameworks used in examples. To do this, execute the following command (it is recommended to use a virtualized Python environment):"

```bash
pip install -r requirements.txt
```

### Example of use

#### `main.py`

The `main.py` file contains the primary execution code. It should import RAMPAGE and create the dataset manager class, which must inherit from `RAMPAGE.DatasetManager`.

```python
from RAMPAGE.Framework import Framework
from PersonaldatasetManager import PersonalDatasetManager
```

Next, the RAMPAGE class and the dataset manager class are instantiated. The dataset manager is then defined within RAMPAGE, and the datasets to be used are added.

```python
# Create Framework
framework = Framework()
# Create DatasetManager1 implementation
datasetManager = PersonalDatasetManager()
# Set percentajes to use in train, validation and test
datasetManager.setPercentages(70,15,15)
# Set in framework the dataset to use
framework.defineDatasetManager(datasetManager)
# Add dga dataset
framework.addDataset(PATH_DGA, True)
# Add Non dga dataset
framework.addDataset(PATH_NON_DGA, True)
```

The next step is to define the classifiers, train, and test them.

```python
# Define classifiers
classifiers = [
    LSTM_example,
    CNN_example,
    Baseline_example
]
for classifier in classifiers:
    classifierObj = classifier()
    framework.addClassifier(classifierObj)
# Train all classifiers
framework.train()
# Test all classifiers
framework.test()
```

After executing the classifier tests, we can retrieve and display the results.

```python
# Get results
results = framework.getResults()
# Show results
for i in range(len(results)):
    print(framework.getClassifierByIndex(i).__class__.__name__)
    print(framework.getResultByIndex(i).toString())
```

#### DatasetManager

TODO

#### DataElement

TODO

#### Result

TODO

#### Classifier

The classifiers inherit from the `RAMPAGE.Classifier` class. To do so, they must implement the train and test functions. A proposed implementation could be as follows:

```python
class Baseline_example(Classifier):

    #Define your configuration values
    ...

    def __init__(self) -> None:
        # Define your model
        ...
        
    def train(self, train:set, validation:set):
        # Define your training method
        ...


    def test(self, test:set) -> Result:
        ...
        return PersonalResult(...)
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