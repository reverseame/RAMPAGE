from typing import List, Tuple, Union

class Result:
    """
    A class to handle result operations with string and CSV format conversions.
    """
    def __init__(self) -> None:
        """
        Initialize Result with empty metrics list.
        The metrics list will store tuples of (name, value) pairs.
        """
        self._metrics: List[Tuple[str, Union[float, int]]] = []

    def add_metric(self, name: str, value: Union[float, int]) -> None:
        """
        Add a metric to the metrics list.

        Args:
            name (str): Name of the metric
            value (Union[float, int]): Value of the metric
        """
        self._metrics.append((name, value))

    def get_metrics(self) -> List[Tuple[str, Union[float, int]]]:
        """
        Get all metrics.

        Returns:
            List[Tuple[str, Union[float, int]]]: List of metric tuples
        """
        return self._metrics

    def __str__(self) -> str:
        """
        Return a string representation of the result.

        Returns:
            str: String representation of the result.
        """
        return "\n".join(f" * {name:<10} -> {value}" for name, value in self._metrics)

    def get_csv_header(self, separator: str) -> str:
        """
        Generate the CSV header string.

        Args:
            separator (str): The delimiter to use between CSV fields.

        Returns:
            str: CSV header string with the specified separator.
        """
        return separator.join(name.lower() for name, _ in self._metrics)

    def to_csv(self, separator: str) -> str:
        """
        Convert the result to CSV format.

        Args:
            separator (str): The delimiter to use between CSV fields.

        Returns:
            str: CSV formatted string with the specified separator.
        """
        return separator.join(str(value) for _, value in self._metrics)