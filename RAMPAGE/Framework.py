import warnings
from RAMPAGE.Classifier import Classifier
from RAMPAGE.Result import Result
from RAMPAGE.DatasetManager import DatasetManager


# Error and warning message templates
WRONG_INDEX_CLASSIFIERS = """
ERROR:

Index out of bounds...

Number of classifiers: {total}
Possible index values: 0 - {max_index}
Index: {current_index}
"""

WRONG_INDEX_RESULTS = """
ERROR:

Index out of bounds...

Number of results: {total}
Possible index values: 0 - {max_index}
Index: {current_index}
"""

WARNING_CLASSIFIER_NOT_FOUND = """
WARNING:

Classifier does not exist
"""

ERROR_NO_DEBUG = """
ERROR:

Debug mode is NOT activated

Not allowed to execute {function_name} without debug mode
"""

class Framework:
    """
    A framework for managing machine learning classifiers and datasets.
    
    Attributes:
        debug (bool): Debug mode flag.
        dataset_manager (DatasetManager): Manager for handling datasets.
        classifiers (list[Classifier]): List of classifiers.
        results (list[Result]): List of results.
    """

    def __init__(self, debug_mode: bool = False) -> None:
        """
        Initialize the framework.

        Args:
            debug_mode (bool, optional): Whether to enable debug mode. Defaults to False.
        """
        self.debug = debug_mode
        self.dataset_manager = None
        self.classifiers = []
        self.results = []

        if self.debug:
            print("\n#############################################")
            print("########### DEBUG MODE ACTIVATED ############")
            print("#############################################\n")

    def set_dataset_manager(self, dataset_manager: DatasetManager) -> None:
        """
        Set the dataset manager instance.

        Args:
            dataset_manager (DatasetManager): The dataset manager to use.
        """
        self.dataset_manager = dataset_manager

    def add_dataset(self, path: str, random_sets: bool) -> None:
        """
        Add and split dataset from file.

        Args:
            path (str): Path to dataset file.
            random_sets (bool): Whether to randomize the splits.
        """
        self.dataset_manager.add(path, random_sets)
        
        if self.debug:
            print("#############################################")
            print("###### NEW DataElements added to sets #######")
            print("#############################################\n")
            print(f"  new size of TRAIN set     : {len(self.dataset_manager.get_train())}")
            print(f"  new size of VALIDATION set: {len(self.dataset_manager.get_validation())}")
            print(f"  new size of TEST set      : {len(self.dataset_manager.get_test())}\n")

    def add_train_dataset(self, path: str) -> None:
        """
        Add data to training set.

        Args:
            path (str): Path to training dataset file.
        """
        self.dataset_manager.add_train(path)

    def add_validation_dataset(self, path: str) -> None:
        """
        Add data to validation set.

        Args:
            path (str): Path to validation dataset file.
        """
        self.dataset_manager.add_validation(path)

    def add_test_dataset(self, path: str) -> None:
        """
        Add data to test set.

        Args:
            path (str): Path to test dataset file.
        """
        self.dataset_manager.add_test(path)

    def add_classifier(self, classifier: Classifier) -> None:
        """
        Add a classifier to the framework.

        Args:
            classifier (Classifier): The classifier to add.
        """
        self.classifiers.append(classifier)
        self.results.append(None)

    def get_classifier_by_index(self, index: int) -> Classifier:
        """
        Get classifier at specified index.

        Args:
            index (int): Index of the classifier to retrieve.

        Returns:
            Classifier: The classifier at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        self._validate_classifier_index(index)
        return self.classifiers[index]

    def clear_classifiers(self) -> None:
        """Clear all classifiers and results."""
        self.classifiers.clear()
        self.results.clear()

    def run(self) -> None:
        """Train and test all classifiers."""
        for i in range(len(self.classifiers)):
            self.run_by_index(i)

    def run_classifier(self, classifier: Classifier) -> None:
        """
        Train and test a specific classifier.

        Args:
            classifier (Classifier): The classifier to run.
        """
        index = self._get_classifier_index(classifier)
        if index is not None:
            self.run_by_index(index)

    def run_by_index(self, index: int) -> None:
        """
        Train and test classifier at specified index.

        Args:
            index (int): Index of the classifier to run.

        Raises:
            IndexError: If index is out of bounds.
        """
        self._validate_classifier_index(index)
        self.train_by_index(index)
        self.test_by_index(index)

    def train(self) -> None:
        """Train all classifiers."""
        for i in range(len(self.classifiers)):
            self.train_by_index(i)

    def train_classifier(self, classifier: Classifier) -> None:
        """
        Train a specific classifier.

        Args:
            classifier (Classifier): The classifier to train.
        """
        index = self._get_classifier_index(classifier)
        if index is not None:
            self.train_by_index(index)

    def train_by_index(self, index: int) -> None:
        """
        Train classifier at specified index.

        Args:
            index (int): Index of the classifier to train.

        Raises:
            IndexError: If index is out of bounds.
        """
        self._validate_classifier_index(index)
        self.classifiers[index].train(
            self.dataset_manager.get_train(),
            self.dataset_manager.get_validation()
        )

    def test(self) -> None:
        """Test all classifiers."""
        for i in range(len(self.classifiers)):
            self.test_by_index(i)

    def test_classifier(self, classifier: Classifier) -> None:
        """
        Test a specific classifier.

        Args:
            classifier (Classifier): The classifier to test.
        """
        index = self._get_classifier_index(classifier)
        if index is not None:
            self.test_by_index(index)

    def test_by_index(self, index: int) -> None:
        """
        Test classifier at specified index.

        Args:
            index (int): Index of the classifier to test.

        Raises:
            IndexError: If index is out of bounds.
        """
        self._validate_classifier_index(index)
        self.results[index] = self.classifiers[index].test(
            self.dataset_manager.get_test()
        )

    def get_results(self) -> list[Result]:
        """
        Get all results.

        Returns:
            list[Result]: List of all classifier results.
        """
        return self.results

    def get_result_for_classifier(self, classifier: Classifier) -> Result:
        """
        Get result for a specific classifier.

        Args:
            classifier (Classifier): The classifier to get results for.

        Returns:
            Result: The result for the classifier, or None if not found.
        """
        index = self._get_classifier_index(classifier)
        return self.get_result_by_index(index) if index is not None else None

    def get_result_by_index(self, index: int) -> Result:
        """
        Get result at specified index.

        Args:
            index (int): Index of the result to retrieve.

        Returns:
            Result: The result at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        self._validate_result_index(index)
        return self.results[index]

    def print_all_dataset(self) -> None:
        """
        Print all datasets (debug mode only).

        Raises:
            SystemExit: If not in debug mode.
        """
        if not self.debug:
            print(ERROR_NO_DEBUG.format(function_name="print_all_dataset()"))
            exit(1)

        for dataset_name, dataset in [
            ("TRAIN", self.dataset_manager.get_train()),
            ("VALIDATION", self.dataset_manager.get_validation()),
            ("TEST", self.dataset_manager.get_test())
        ]:
            print("#############################################")
            print(f"############## {dataset_name} SET LIST ###############")
            print("#############################################\n")
            
            for element in dataset:
                print(f"  - {element.domain} -> {element.is_dga}")
            print()

    def _validate_classifier_index(self, index: int) -> None:
        """
        Validate classifier index.

        Args:
            index (int): Index to validate.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < len(self.classifiers):
            raise IndexError(WRONG_INDEX_CLASSIFIERS.format(
                total=len(self.classifiers),
                max_index=len(self.classifiers) - 1,
                current_index=index
            ))

    def _validate_result_index(self, index: int) -> None:
        """
        Validate result index.

        Args:
            index (int): Index to validate.

        Raises:
            IndexError: If index is out of bounds.
        """
        if not 0 <= index < len(self.results):
            raise IndexError(WRONG_INDEX_RESULTS.format(
                total=len(self.results),
                max_index=len(self.results) - 1,
                current_index=index
            ))

    def _get_classifier_index(self, classifier: Classifier) -> int:
        """
        Get index of classifier or None if not found.

        Args:
            classifier (Classifier): The classifier to find.

        Returns:
            int: Index of the classifier, or None if not found.
        """
        try:
            return self.classifiers.index(classifier)
        except ValueError:
            warnings.warn(WARNING_CLASSIFIER_NOT_FOUND)
            return None