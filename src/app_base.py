# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class Parameters(metaclass=ABCMeta):
    def __init__(self, parser):
        #params = parser.parse_args()

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

    def get_title_from_params(self):

        members = [attr for attr in vars(self).items() ]

        param_str = []
        param_dict = {}
        used_words = []
        for member in members:
            name = member[0]
            value = member[1]

            param_name = ''
            for c in name:
                param_name += c
                if param_name not in used_words:
                    break

            used_words.append(param_name)
            param_dict.update({param_name : value})

            param_str.append( param_name + str(value) )

        return "-".join(param_str), param_dict



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
    def prepare_data(self, csv_data):
        # Make training data from csv_data
        pass

    @abstractmethod
    def get_rawdata_and_labels(self, rawdata):
        # Make csv data from rawdata and labels
        pass

    @abstractmethod
    def is_alive(self):
        return self.is_exit != True

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
    def prepare_data(self, rawdata):
        pass

    @abstractmethod
    def set_predict_result(self, predicted):
        self.predicted = predicted
        pass

    @abstractmethod
    def is_alive(self):
        return self.is_exit != True

class DataAugmentation(metaclass=ABCMeta):
    def __init__(self, parameters=[]):
        self.parameters = parameters

    @abstractmethod
    def get_augmented_data(self, pulse):
        data = []
        return data
