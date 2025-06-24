# RAMPAGE - A Training and Comparing AGD Detectors Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Version: v1.1](https://img.shields.io/badge/Version-v1.1.0-green.svg)]()

RAMPAGE: (*fRAMework to comPAre aGd dEtectors*) is a framework aimed at training and comparing machine learning models for the detection of Algorithmically Generated Domains (AGDs).

## Features

* Training and testing of machine learning models for detecting AGDs
* A structured, clean and straightforward approach to comparing machine learning models
* Possibility of extending the base data types to add new features to dataset elements
* Automated management of datasets
* Simple way to share models and results

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

The `main.py` file contains the primary execution code. It should import RAMPAGE `Framework` and `DatasetManager` class.

```python
from RAMPAGE.Framework import Framework
from RAMPAGE.DatasetManager import DatasetManager
```

Next, `DatasetManager` is then defined within RAMPAGE `Framework`, and the datasets to be used are added.

```python
# Create Framework
framework = Framework()
# Create DatasetManager1 implementation
datasetManager = DatasetManager()
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
    LSTMExample,
    CNNExample,
    BaselineExample
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

#### Datasets Definition

For managing datasets, two classes need to be considered: `DataElement` and `DatasetManager`. `DataElement` represents a single unit with all its features. In the base version, it only includes the domain and a boolean indicating whether the domain should be classified as malicious or not. If new fields or features need to be added, two new classes must be created, inheriting from `DataElement` and `DatasetManager`, respectively.

In `DataElement`, you need to add as many attributes to the class as the number of features you want to include. In `DatasetManager`, the `parse_data_element` function must be overridden so that it can read the new fields of the updated `DataElement`.

Base `DatasetManager` follows `<domain>;<"True"/"False">` syntax (without `<` and `>` characters).

#### Result

`Result` is empty by default. Therefore, a new class that inherits from `Result` should be created, where the desired metrics for the statistics to be measured will be implemented. E.g.:


```python
class ResultPersonal(Result):

    def __init__(self, accuracy, precision, recall):
        super().__init__()

        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
```

#### Classifier

The classifiers inherit from the RAMPAGE `Classifier` class. To do so, they must implement the train and test functions. A proposed implementation could be as follows:

```python
class Example(Classifier):

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
@article{pelayobenedetrampage,
title = {RAMPAGE: a software framework to ensure reproducibility in algorithmically generated domains detection},
journal = {Expert Systems with Applications},
volume = {293},
pages = {128629},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.128629},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425022481},
author = {Tomás Pelayo-Benedet and Ricardo J. Rodríguez and Carlos H. Gañán}
```

## Funding support

Part of this research was supported by the Spanish National Cybersecurity Institute (INCIBE) under *Proyectos Estratégicos de Ciberseguridad -- CIBERSEGURIDAD EINA UNIZAR* and by the Recovery, Transformation and Resilience Plan funds, financed by the European Union (Next Generation).

![INCIBE_logos](misc/img/INCIBE_logos.jpg)