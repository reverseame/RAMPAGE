class Result:
    """
    A class to handle result operations with string and CSV format conversions.
    """

    def __str__(self) -> str:
        """
        Return a string representation of the result.

        Returns:
            str: String representation of the result.
        """
        pass

    def get_csv_header(self, separator: str) -> str:
        """
        Generate the CSV header string.

        Args:
            separator (str): The delimiter to use between CSV fields.

        Returns:
            str: CSV header string with the specified separator.
        """
        pass

    def to_csv(self, separator: str) -> str:
        """
        Convert the result to CSV format.

        Args:
            separator (str): The delimiter to use between CSV fields.

        Returns:
            str: CSV formatted string with the specified separator.
        """
        pass