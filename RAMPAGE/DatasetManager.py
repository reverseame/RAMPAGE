import warnings
import random
from RAMPAGE.DataElement import DataElement


# Error and warning message templates
WRONG_PERCENTAGES_MESSAGE = """ERROR:

Wrong percentages of train, validation and test...

train + validation + test =
{train_pct} + {validation_pct} + {test_pct} =
{total_pct}
"""

WARNING_PERCENTAGES_MESSAGE = """WARNING:

Train percentage should be greater than validation percentage...

train <= validation
{train_pct} <= {validation_pct}
"""


class DatasetManager:
    """
    A class to manage dataset splitting and loading for machine learning tasks.
    
    This class handles the division of data into training, validation, and test sets,
    with configurable percentages and optional randomization.
    """

    def __init__(self):
        """Initialize DatasetManager with default settings."""
        self.train_set = set()
        self.validation_set = set()
        self.test_set = set()
        self.train_pct = 80
        self.validation_pct = 10
        self.test_pct = 10

    def set_percentages(self, train: int, validation: int, test: int) -> None:
        """
        Set split percentages for train, validation and test sets.
        
        Args:
            train (int): Percentage of data for training.
            validation (int): Percentage of data for validation.
            test (int): Percentage of data for testing.
            
        Raises:
            Exception: If percentages don't sum to 100.
        """
        if train + validation + test != 100:
            raise Exception(WRONG_PERCENTAGES_MESSAGE.format(
                train_pct=train,
                validation_pct=validation,
                test_pct=test,
                total_pct=train + validation + test
            ))

        if train <= validation:
            warnings.warn(WARNING_PERCENTAGES_MESSAGE.format(
                train_pct=train,
                validation_pct=validation
            ))
        
        self.train_pct = train
        self.validation_pct = validation
        self.test_pct = test

    def get_train(self) -> set:
        """Return the training set."""
        return self.train_set

    def get_validation(self) -> set:
        """Return the validation set."""
        return self.validation_set

    def get_test(self) -> set:
        """Return the test set."""
        return self.test_set
    
    def add(self, path: str, random_sets: bool) -> None:
        """
        Load and split data from file into train, validation and test sets.
        
        Args:
            path (str): Path to the data file.
            random_sets (bool): Whether to randomize the data split.
        """
        data_elements = []
        with open(path, 'r') as f:
            data_elements = [self.parse_data_element(line) for line in f]

        if random_sets:
            random.shuffle(data_elements)

        total_elements = len(data_elements)
        train_end = int(total_elements * self.train_pct / 100)
        val_end = int(total_elements * (self.train_pct + self.validation_pct) / 100)

        self.train_set.update(data_elements[:train_end])
        self.validation_set.update(data_elements[train_end:val_end])
        self.test_set.update(data_elements[val_end:])

    def add_train(self, path: str) -> None:
        """
        Add data from file to training set.
        
        Args:
            path (str): Path to the data file.
        """
        with open(path, 'r') as f:
            self.train_set.update(self.parse_data_element(line) for line in f)

    def add_validation(self, path: str) -> None:
        """
        Add data from file to validation set.
        
        Args:
            path (str): Path to the data file.
        """
        with open(path, 'r') as f:
            self.validation_set.update(self.parse_data_element(line) for line in f)

    def add_test(self, path: str) -> None:
        """
        Add data from file to test set.
        
        Args:
            path (str): Path to the data file.
        """
        with open(path, 'r') as f:
            self.test_set.update(self.parse_data_element(line) for line in f)

    def clear(self) -> None:
        """Clear all data sets."""
        self.train_set.clear()
        self.validation_set.clear()
        self.test_set.clear()

    def parse_data_element(self, line: str) -> DataElement:
        """
        Parse a line of text into a DataElement.
        
        Args:
            line (str): Line of text to parse.
            
        Returns:
            DataElement: Parsed data element.
        """
        domain, is_dga_str = line.strip().split(";")
        is_dga = bool(eval(is_dga_str))
        return DataElement(domain, is_dga)