from abc import abstractmethod


class Detector:
    @abstractmethod
    def detect(self, fd):
        """
        Processes the file and return a result and costs - the probability 
        for the file being malicious (ranged from 0 to 1) and costs of any 
        type (time, CPU, money, etc.). If the detector does not return a 
        probability, it can return boolean result: 0 or 1.
        
        The process can be eith a classic ML process (feature extraction and
        prediction), or scanning the file through an existing product and 
        processing the result.
        """
        result = None
        costs = None
        return result, costs
