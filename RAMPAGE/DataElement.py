################################################################################
#  CODE  #######################################################################
################################################################################

class DataElement:

    domain:str
    isDGA:bool

    def __init__(self, domain:str, isDGA:bool) -> None:
        self.domain = domain
        self.isDGA = isDGA

    def __iter__(self):
        return iter(self)