# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import threading

class DataStreamerBase(threading.Thread, metaclass=ABCMeta):
    def __init__(self):
        threading.Thread.__init__(self)
        self.callback = None
        self.is_exit = False

    @abstractmethod
    def set_callback(self, callback):
        self.callback = callback

    @abstractmethod
    def run(self):
        self.start()
        pass

    @abstractmethod
    def stop(self):
        self.is_exit = True
        pass
