from RAMPAGE.Result import Result

class Classifier:
    """
    A base classifier class that provides interface for training and testing.
    
    This class serves as a template for implementing different classification algorithms.
    """
    
    def train(self, train_set: set, validation_set: set) -> None:
        """
        Train the classifier using training and validation datasets.
        
        Args:
            train_set (set): The set of training data.
            validation_set (set): The set of validation data.
        """
        pass
    
    def test(self, test_set: set) -> Result:
        """
        Test the trained classifier on a test dataset.
        
        Args:
            test_set (set): The set of test data.
            
        Returns:
            Result: The classification results.
        """
        pass