from RAMPAGE.DatasetManager import DatasetManager
from RAMPAGE.DataElement import DataElement

################################################################################
#  CODE  #######################################################################
################################################################################

class DatasetManagerCommon(DatasetManager):

    def parseDataElement(self, line:str) -> DataElement:
        domain = line.split(";")[0]
        isDGAstr = line.split(";")[1]

        isDGA = False
        if (eval(isDGAstr)):
            isDGA = True

        return DataElement(domain, isDGA)