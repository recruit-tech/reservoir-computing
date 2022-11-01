# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import threading

class DataStreamerBase(threading.Thread, metaclass=ABCMeta):
    def __init__(self):
        threading.Thread.__init__(self)
        self.callback = None
        self.is_exit = False
        self.start()

    def begin(self, callback):
        self.callback = callback

    def end(self):
        self.callback = None

    def kill(self):
        self.is_exit = True

    @abstractmethod
    def run(self):
        pass
