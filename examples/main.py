from RAMPAGE.Framework import Framework
from RAMPAGE.DatasetManager import DatasetManager

from classifiers.LSTM import LSTMExample
from classifiers.CNN import CNNExample
from classifiers.baseline import BaselineExample


def main():
    """
    Main function to run the DGA detection framework with multiple classifiers.
    """
    # Define dataset paths
    # TODO: Replace with actual paths
    PATH_DGA = "***"
    PATH_NON_DGA = "***"

    # Initialize and configure framework
    framework = setup_framework()
    
    # Load and process datasets
    load_datasets(framework, PATH_DGA, PATH_NON_DGA)
    
    # Train and evaluate classifiers
    run_classifiers(framework)
    
    # Print results
    print_results(framework)


def setup_framework() -> Framework:
    """
    Initialize and configure the framework and dataset manager.
    
    Returns:
        Framework: Configured framework instance.
    """
    # Create framework instance
    framework = Framework()
    
    # Configure dataset manager
    dataset_manager = DatasetManager()
    dataset_manager.set_percentages(
        train=70,
        validation=15,
        test=15
    )
    
    # Set dataset manager in framework
    framework.set_dataset_manager(dataset_manager)
    
    return framework


def load_datasets(framework: Framework, dga_path: str, non_dga_path: str) -> None:
    """
    Load DGA and non-DGA datasets into the framework.
    
    Args:
        framework: The framework instance.
        dga_path: Path to DGA dataset.
        non_dga_path: Path to non-DGA dataset.
    """
    # Add datasets with randomization enabled
    framework.add_dataset(dga_path, random_sets=True)
    framework.add_dataset(non_dga_path, random_sets=True)


def run_classifiers(framework: Framework) -> None:
    """
    Initialize, train and test all classifiers.
    
    Args:
        framework: The framework instance.
    """
    # Define classifier classes
    classifier_classes = [
        LSTMExample,
        CNNExample,
        BaselineExample
    ]
    
    # Initialize and add classifiers to framework
    for classifier_class in classifier_classes:
        classifier = classifier_class()
        framework.add_classifier(classifier)
    
    # Train and test all classifiers
    framework.train()
    framework.test()


def print_results(framework: Framework) -> None:
    """
    Print classification results for all classifiers.
    
    Args:
        framework: The framework instance.
    """
    results = framework.get_results()
    
    print("\n=== Classification Results ===\n")
    
    for i in range(len(results)):
        classifier = framework.get_classifier_by_index(i)
        result = framework.get_result_by_index(i)
        
        print(f"\nClassifier: {classifier.__class__.__name__}")
        print("-" * 40)
        print(result)
        print()


if __name__ == "__main__":
    main()