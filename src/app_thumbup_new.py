# -*- coding: utf-8 -*-
import app_base
import numpy as np


###############################################################################
# Common
###############################################################################
class Parameters(app_base.Parameters):
    def __init__(self, parser):
        super().__init__(parser)
        #self.set_grid_search_params_list()
        self._idx = 0
        self._num_of_grid_search = 0

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

    def set_next_grid_search_params(self):
        if self._idx >= self._num_of_grid_search and self._idx != 0:
            return False

        if self._idx == 0:
            self._grid_search_params_list = []
            for node in [800,900,1000]:
                for density in [0.4,]:
                    for input_scale in [0.004,]:
                        for rho in [1.0,]:
                            for fb_scale in [None,]:
                                for leaking_rate in [0.1,]:
                                    for average_window in [1,]:
            #for node in [700,800,900]:
            #    for density in [0.1,0.2,0.4,0,8,1.0]:
            #        for input_scale in [0.001,0.002,0.004,0.008,0.016]:
            #            for rho in [0.6,0.7,0.8,0.9,1.0]:
            #                for fb_scale in [None,]:
            #                    for leaking_rate in [0.05,0.1,0.2,0.4,0.8]:
            #                        for average_window in [1,2,4,8,16]:
                                        self._grid_search_params_list.append([node,density,input_scale,rho,fb_scale,leaking_rate,average_window])
                                        self._num_of_grid_search += 1

        params = self._grid_search_params_list[self._idx]
        self.node = params[0]
        self.density = params[1]
        self.input_scale = params[2]
        self.rho = params[3]
        self.fb_scale = params[4]
        self.leaking_rate = params[5]
        self.average_window = params[6]

        self._idx += 1
        return True
        

    def get_title_from_params(self):

        title = ['n' + f'{self.node:04}', \
                 'd' + str('{:.3f}'.format(self.density)), \
                 'i' + str('{:.3f}'.format(self.input_scale)), \
                 'r' + str('{:.3f}'.format(self.rho)), \
                 'f' + str('{}'.format(self.fb_scale)), \
                 'l' + str('{:.3f}'.format(self.leaking_rate)), \
                 'w' + f'{self.average_window:03}' ]

        return "-".join(title)


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
        pygame.init()
        self.width = 151
        self.height = 181 + 50
        self.display = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("Arial", 35)

        white = (230, 230, 230)
        self.title = self.font.render('Training', True, white)
        self.title_rect = self.title.get_rect(topright = (110, 185))

        self.bg_thumb_neutral = pygame.image.load("thumb_neutral.png")
        self.bg_thumb_up = pygame.image.load("thumb_up.png")
        self.bg = self.bg_thumb_neutral
        self.E_THUMB_NEUTRAL = pygame.event.Event(pygame.USEREVENT, attr1='E_THUMB_NEUTRAL')
        self.E_THUMB_UP      = pygame.event.Event(pygame.USEREVENT, attr1='E_THUMB_UP')
        pygame.display.set_caption("Reservoir Computing - thumb up detection")
        self.clock = pygame.time.Clock()
        #self.data0 = DataAugmentation(self.parametes)
        self.is_thumb_neutral = True

        super().start()
        loop = True
        while loop:
            for event in pygame.event.get():
                ''' End the game only when the 'quit' button is pressed '''
                if event.type == pygame.QUIT:
                    print('stopped TrainingApp')
                    pygame.quit()
                    self.is_exit = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print('stopped TrainingApp')
                        pygame.quit()
                        self.is_exit = True
                        return

                if event == self.E_THUMB_NEUTRAL:
                    self.bg = self.bg_thumb_neutral

                if event == self.E_THUMB_UP:
                    self.bg = self.bg_thumb_up

            self.display.blit(self.bg, (0, 0))
            self.display.blit(self.title, self.title_rect)

            pygame.display.update()
            self.clock.tick(60)

            if self.is_exit:
                print('stopped TrainingApp')
                pygame.quit()
                return

    def stop(self):
        super().stop()
        

    #def close(self):
    #    pygame.quit()
    #    #sys.exit()
    
    def get_rawdata_and_labels(self, rawdata):
        csv_data = rawdata
        return csv_data

    def get_data(self, csv_data):
        pulse00 = csv_data[0]
        pulse01 = csv_data[1]
        pulse02 = csv_data[2]
        buttons = csv_data[3]
        return pulse00, buttons

    def prepare_data(self, csv_data):
        pulse00, label = self.get_data(csv_data)
        pulses = [float(pulse00),]
        labels = [float(int(label) == 1), ]

        if pygame.get_init() == True:
            if int(label) == 2:
                if self.is_thumb_neutral == True:
                    pygame.event.post(self.E_THUMB_UP)
                    self.is_thumb_neutral = False
            else:
                if self.is_thumb_neutral != True:
                    pygame.event.post(self.E_THUMB_NEUTRAL)
                    self.is_thumb_neutral = True

        return pulses, labels

    def is_alive(self):
        return super().is_alive()


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
        pygame.init()
        self.width = 151
        self.height = 181 + 50
        self.display = pygame.display.set_mode((self.width, self.height))

        self.font = pygame.font.SysFont("Arial", 35)
        white = (230, 230, 230)
        self.title = self.font.render('Predict', True, white)
        self.title_rect = self.title.get_rect(topright = (110, 185))
        self.bg_thumb_neutral = pygame.image.load("thumb_neutral.png")
        self.bg_thumb_up = pygame.image.load("thumb_up.png")
        self.bg = self.bg_thumb_neutral
        self.E_THUMB_NEUTRAL = pygame.event.Event(pygame.USEREVENT, attr1='E_THUMB_NEUTRAL')
        self.E_THUMB_UP      = pygame.event.Event(pygame.USEREVENT, attr1='E_THUMB_UP')
        pygame.display.set_caption("Reservoir Computing - thumb up detection")
        self.clock = pygame.time.Clock()
        #self.data0 = DataAugmentation(self.parametes)
        self.is_thumb_neutral = True
        self.moving_avg_win_size = 4

    def start(self):
        super().start()
        loop = True
        while loop:
            for event in pygame.event.get():
                ''' End the game only when the 'quit' button is pressed '''
                if event.type == pygame.QUIT:
                    print('stopped PredictApp')
                    pygame.quit()
                    self.is_exit = True
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print('stopped PredictApp')
                        pygame.quit()
                        self.is_exit = True
                        return

                if event == self.E_THUMB_NEUTRAL:
                    self.bg = self.bg_thumb_neutral

                if event == self.E_THUMB_UP:
                    self.bg = self.bg_thumb_up

            self.display.blit(self.bg, (0, 0))
            self.display.blit(self.title, self.title_rect)
            pygame.display.update()
            self.clock.tick(60)

            if self.is_exit:
                print('stopped PredictApp')
                pygame.quit()
                return

    def stop(self):
        super().stop()

    #def close(self):
    #    pygame.quit()
    #    sys.exit()
    
    def get_data(self, rawdata):
        pulse00 = rawdata[0]
        #pulse01 = rawdata[1]
        #pulse02 = rawdata[2]
        #pulse03 = rawdata[3]
        buttons = rawdata[4]
        return pulse00

    def prepare_data(self, rawdata):
        pulse00 = self.get_data(rawdata)
        pulses = [float(pulse00),]

        return pulses

    def set_predict_result(self, predicted):
        super().set_predict_result(predicted)
        if len(predicted[-self.moving_avg_win_size:-1]) == 0:
            print('P skip02')
            return

        avg = np.mean(predicted[-self.moving_avg_win_size:-1][0][0])

        if avg > 0.5:
            if self.is_thumb_neutral == True:
                pygame.event.post(self.E_THUMB_UP)
                self.is_thumb_neutral = False
        else:
            if self.is_thumb_neutral != True:
                pygame.event.post(self.E_THUMB_NEUTRAL)
                self.is_thumb_neutral = True
        return

    def is_alive(self):
        return super().is_alive()


if __name__=="__main__":
    training_app = TrainingApp()

