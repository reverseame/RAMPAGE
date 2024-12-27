from RAMPAGE.Result import Result
import math

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
        """
        super().__init__()
        
        # Calculate all metrics
        f1_score = self._calculate_f1_score(precision, recall)
        fpr = self._calculate_fpr(fp, tn)
        tpr = self._calculate_tpr(tp, fn)
        mcc = self._calculate_mcc(tp, tn, fp, fn)
        kappa = self._calculate_kappa(tp, tn, fp, fn)
        
        # Add all metrics using the base class method
        metrics = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 score", f1_score),
            ("FPR", fpr),
            ("TPR", tpr),
            ("AUC", auc),
            ("FP", fp),
            ("FN", fn),
            ("TP", tp),
            ("TN", tn),
            ("MCC", mcc),
            ("Kappa", kappa)
        ]
        
        for name, value in metrics:
            self.add_metric(name, value)

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
