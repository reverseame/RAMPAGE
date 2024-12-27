class DataElement:
    """
    A class to represent a domain data element with DGA classification.
    
    Attributes:
        domain (str): The domain name string.
        is_dga (bool): Boolean flag indicating if the domain is DGA (Domain Generation Algorithm).
    """
    
    def __init__(self, domain: str, is_dga: bool) -> None:
        """
        Initialize a new DataElement instance.
        
        Args:
            domain (str): The domain name to store.
            is_dga (bool): Flag indicating if the domain is DGA.
        """
        self.domain = domain
        self.is_dga = is_dga
    
    def __iter__(self):
        """
        Make the class iterable.
        
        Returns:
            iterator: An iterator of the instance.
        """
        return iter(self)