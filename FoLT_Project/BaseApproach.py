import abc
import logging


class BaseApproach(object):
    def __init__(self):
        logging.info("Approach created")

    @abc.abstractmethod
    def run(self):
        return
