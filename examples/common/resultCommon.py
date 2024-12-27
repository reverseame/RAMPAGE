from RAMPAGE.Result import Result
import math
from typing import Union


class ResultCommon(Result):
    """
    A class to store and calculate common classification metrics.
    
    This class extends the base Result class and provides implementation
    for various classification metrics including accuracy, precision,
    recall, F1-score, TPR, FPR, AUC, MCC, and Kappa coefficient.
    """

    def __init__(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        fp: int,
        fn: int,
        tp: int,
        tn: int,
        auc: float
    ) -> None:
        """
        Initialize ResultCommon with classification metrics.

        Args:
            accuracy (float): Classification accuracy.
            precision (float): Classification precision.
            recall (float): Classification recall.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.
            tp (int): Number of true positives.
            tn (int): Number of true negatives.
            auc (float): Area Under the Curve value.
        """
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        
        # Calculate F1 score
        self.f1_score = self._calculate_f1_score(precision, recall)
        
        # Calculate False Positive Rate and True Positive Rate
        self.fpr = self._calculate_fpr(fp, tn)
        self.tpr = self._calculate_tpr(tp, fn)
        
        # Store basic metrics
        self.fp = fp
        self.fn = fn
        self.tp = tp
        self.tn = tn
        self.auc = auc
        
        # Calculate Matthews Correlation Coefficient
        self.mcc = self._calculate_mcc(tp, tn, fp, fn)
        
        # Calculate Kappa coefficient
        self.kappa = self._calculate_kappa(tp, tn, fp, fn)

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.

        Args:
            precision (float): Classification precision.
            recall (float): Classification recall.

        Returns:
            float: F1 score value.
        """
        if precision + recall != 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def _calculate_fpr(self, fp: int, tn: int) -> float:
        """
        Calculate False Positive Rate.

        Args:
            fp (int): Number of false positives.
            tn (int): Number of true negatives.

        Returns:
            float: False Positive Rate value.
        """
        denominator = fp + tn
        return fp / denominator if denominator != 0 else 0.0

    def _calculate_tpr(self, tp: int, fn: int) -> float:
        """
        Calculate True Positive Rate.

        Args:
            tp (int): Number of true positives.
            fn (int): Number of false negatives.

        Returns:
            float: True Positive Rate value.
        """
        denominator = tp + fn
        return tp / denominator if denominator != 0 else 0.0

    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calculate Matthews Correlation Coefficient.

        Args:
            tp (int): Number of true positives.
            tn (int): Number of true negatives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.

        Returns:
            float: Matthews Correlation Coefficient value.
        """
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator != 0:
            return ((tp * tn) - (fp * fn)) / denominator
        return 0.0

    def _calculate_kappa(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calculate Cohen's Kappa coefficient.

        Implementation based on:
        https://stackoverflow.com/questions/58735642/how-to-correctly-implement-cohen-kappa-metric-in-keras

        Args:
            tp (int): Number of true positives.
            tn (int): Number of true negatives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.

        Returns:
            float: Kappa coefficient value.
        """
        total = tn + fp + fn + tp
        if total == 0:
            return 0.0
            
        p0 = (tn + tp) / total
        pa = ((tn + tp) / total) * ((tn + fn) / total)
        pb = ((fn + tp) / total) * ((fp + tp) / total)
        pe = pa + pb
        
        return (p0 - pe) / (1 - pe) if pe != 1 else 0.0

    def __str__(self) -> str:
        """
        Convert results to formatted string.

        Returns:
            str: Formatted string with all metrics.
        """
        metrics = [
            ("Accuracy", self.accuracy),
            ("Precision", self.precision),
            ("Recall", self.recall),
            ("F1 score", self.f1_score),
            ("FPR", self.fpr),
            ("TPR", self.tpr),
            ("AUC", self.auc),
            ("MCC", self.mcc),
            ("Kappa", self.kappa)
        ]
        
        return "\n".join(f" * {name:<10} -> {value}" for name, value in metrics)

    def get_csv_header(self, separator: str) -> str:
        """
        Get CSV header string.

        Args:
            separator (str): CSV field separator.

        Returns:
            str: CSV header string.
        """
        fields = [
            "accuracy", "precision", "recall", "f1", "fpr", "tpr",
            "auc", "fp", "fn", "tp", "tn", "mcc", "kappa"
        ]
        return separator.join(fields)

    def to_csv(self, separator: str) -> str:
        """
        Convert results to CSV format.

        Args:
            separator (str): CSV field separator.

        Returns:
            str: CSV formatted string with all metrics.
        """
        values = [
            self.accuracy, self.precision, self.recall,
            self.f1_score, self.fpr, self.tpr, self.auc,
            self.fp, self.fn, self.tp, self.tn,
            self.mcc, self.kappa
        ]
        return separator.join(str(value) for value in values)