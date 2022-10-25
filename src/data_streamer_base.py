# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import threading

class DataStreamerBase(threading.Thread, metaclass=ABCMeta):
    def __init__(self):
        threading.Thread.__init__(self)
        self.started = threading.Event()

        self.callback = None
        self.is_exit = False
        self.start()

    def __del__(self):
        self.kill()

    def begin(self):
        self.is_exit = False
        self.started.set()

    def end(self):
        self.is_exit = True
        self.started.clear()

    def kill(self):
        self.is_exit = True
        self.started.set()
        self.join()

    @abstractmethod
    def set_callback(self, callback):
        self.callback = callback

    @abstractmethod
    def run(self):
        pass

    #@abstractmethod
    #def stop(self):
    #    self.is_exit = True
    #    pass

