from tensorflow.keras.metrics import (
    Precision,
    Recall,
    FalsePositives,
    FalseNegatives,
    TruePositives,
    TrueNegatives,
    AUC
)


class CommonData:
    """
    A class to store common configuration parameters for neural network models.

    This class defines default values for training parameters and metrics
    used across different models in the framework.

    Attributes:
        epochs (int): Number of training epochs.
        max_length (int): Maximum length for domain name sequences.
        batch_size (int): Size of batches for training.
        verbose (int): Verbosity level for training output (0: silent, 1: progress bar, 2: one line per epoch).
        metrics (list): List of metrics to track during training and evaluation.
    """

    def __init__(self) -> None:
        """Initialize CommonData with default configuration values."""
        # Training parameters
        self.epochs = 1
        self.max_length = 70  # Maximum domain name length
        self.batch_size = 50
        self.verbose = 1

        # Metrics configuration
        self.metrics = [
            'accuracy',
            Precision(),
            Recall(),
            FalsePositives(),
            FalseNegatives(),
            TruePositives(),
            TrueNegatives(),
            AUC()
        ]