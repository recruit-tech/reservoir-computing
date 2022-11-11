# -*- coding: utf-8 -*-
import app_base
from matplotlib import pyplot as plt, animation as an
import numpy as np
import matplotlib.patches as patches
import sys

###############################################################################
# Common
###############################################################################
class Parameters(app_base.Parameters):
    def __init__(self, parser):
        super().__init__(parser)

    def add_hyper_parameters(self, parser):
        # Hyper parametes for reserver computing 
        parser.add_argument('-node', dest='node', default=800, type=int, help='number of node')
        parser.add_argument('-density', dest='density', default=0.2, type=float, help='density')
        parser.add_argument('-input_scale', dest='input_scale', default=0.004, type=float, help='input scale')
        parser.add_argument('-rho', dest='rho', default=0.999, type=float, help='rho')
        parser.add_argument('-fb_scale', dest='fb_scale', default=None, type=float, help='fb scale')
        parser.add_argument('-leaking_rate', dest='leaking_rate', default=0.1, type=float, help='leaking rate')
        parser.add_argument('-average_window', dest='average_window', default=1, type=int, help='average window size')
        parser.add_argument('--no_classification', dest='no_class', action='store_false', help='no class')


    def add_custome_perametes(self, parser):
        parser.add_argument('-num_of_input_data', dest='num_of_input_data', default=1, type=int, help='num of input data (num of sensors)')
        parser.add_argument('-num_of_augmented_data', dest='num_of_augmented_data', default=1, type=int, help='num of input data (num of sensors)')
        parser.add_argument('-num_of_output_classes', dest='num_of_output_classes', default=1, type=int, help='num of output claasses')
        parser.add_argument('-training_time_in_sec', dest='training_time_in_sec', default=60, type=int, help='Training time in sec')


    def set_parameters(self, params):
        self.node = params.node
        self.density = params.density
        self.input_scale = params.input_scale
        self.rho = params.rho
        self.fb_scale = params.fb_scale
        self.leaking_rate = params.leaking_rate
        self.average_window = params.average_window
        self.no_class = params.no_class

        self.num_of_input_data = params.num_of_input_data
        self.num_of_augmented_data = params.num_of_augmented_data
        self.num_of_output_classes = params.num_of_output_classes
        self.training_time_in_sec = params.training_time_in_sec



class DataAugmentation(app_base.DataAugmentation):
    def __init__(self, parameters=[]):
        self.parameters = parameters

    def get_augmented_data(self, pulse):
        return pulse


###############################################################################
# Train
###############################################################################
class TrainingApp(app_base.TrainingApp):
    def __init__(self, parametes):
        super().__init__()
        self.parametes = parametes

    def start(self):
        super().start()
        loop = True
        while loop:

            if self.is_exit:
                print('stopped TrainingApp')
                return

    def stop(self):
        super().stop()

    def get_rawdata_and_labels(self, rawdata):
        return rawdata

    def get_data(self, csv_data):
        pulse00 = csv_data[0]
        buttons = csv_data[4]
        return pulse00, buttons

    def prepare_data(self, csv_data):
        pulse00, label = self.get_data(csv_data)
        pulses = [float(pulse00),]
        labels = [float(int(label) == 2), ]
        return pulses, labels

###############################################################################
# Predict
###############################################################################
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import time

class PredictApp(app_base.PredictApp):
    def __init__(self, parametes):
        super().__init__()
        self.parametes = parametes

    def start(self):
        super().start()
        loop = True
        while loop:

            if self.is_exit:
                print('stopped TrainingApp')
                return

    def stop(self):
        super().stop()

    def get_data(self, rawdata):
        pulse00 = rawdata[0]
        #pulse01 = data[1]
        #pulse02 = data[2]
        #pulse03 = data[3]
        buttons = rawdata[4]
        return pulse00

    def prepare_data(self, rawdata):
        pulse00 = self.get_data(rawdata)
        pulses = [float(pulse00),]

        return pulses

    def set_predict_result(self, predicted):
        super().set_predict_result(predicted)
        return

if __name__=="__main__":
    training_app = TrainingApp()

