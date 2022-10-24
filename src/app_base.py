# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class Parameters(metaclass=ABCMeta):
    def __init__(self, parser):
        params = parser.parse_args()

        try:
            self.add_hyper_parameters(parser)
            self.add_custome_perametes(parser)
        except:
            print('Skip add args')
        params = parser.parse_args()
        self.set_parameters(params)

    @abstractmethod
    def add_hyper_parameters(self, parser):
        pass

    @abstractmethod
    def add_custome_perametes(self, parser):
        pass

    @abstractmethod
    def set_parameters(self, params):
        pass


class TrainingApp(metaclass=ABCMeta):
    def __init__(self):
        self.is_exit = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        self.is_exit = True
        pass

    @abstractmethod
    def prepare_data(self, data):
        pass


class PredictApp(metaclass=ABCMeta):
    def __init__(self):
        self.is_exit = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        self.is_exit = True
        pass

    @abstractmethod
    def prepare_data(self, data):
        pass

    @abstractmethod
    def set_predict_result(self, predicted):
        self.predicted = predicted
        pass

class DataAugmentation(metaclass=ABCMeta):
    def __init__(self, parameters=[]):
        self.parameters = parameters

    @abstractmethod
    def get_augmented_data(self, pulse):
        data = []
        return data
