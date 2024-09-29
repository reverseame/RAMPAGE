from RAMPAGE.Framework import Framework

from common.datasetManagerCommon import DatasetManagerCommon

from classifiers.LSTM import LSTM_example
from classifiers.CNN import CNN_example
from classifiers.baseline import Baseline_example

# Define your dataset path
PATH_DGA = "TBD"
PATH_NON_DGA = "TBD"

# Create Framework
framework = Framework()

# Create DatasetManager1 implementation
datasetManager = DatasetManagerCommon()

# Set percentajes to use in train, validation and test
datasetManager.setPercentages(70,15,15)

# Set in framework the dataset to use
framework.defineDatasetManager(datasetManager)

# Add dga dataset
framework.addDataset(PATH_DGA, True)

# Add Non dga dataset
framework.addDataset(PATH_NON_DGA, True)

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

results = framework.getResults()

for i in range(len(results)):
    print(framework.getClassifierByIndex(i).__class__.__name__)
    print(framework.getResultByIndex(i).toString())
