# -*- coding: utf-8 -*-
import app_base
from matplotlib import pyplot as plt, animation as an
import numpy as np
import matplotlib.patches as patches

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
        parser.add_argument('-num_of_augmented_data', dest='num_of_augmented_data', default=15, type=int, help='num of input data (num of sensors)')
        parser.add_argument('-num_of_output_classes', dest='num_of_output_classes', default=6, type=int, help='num of output claasses')
        parser.add_argument('-training_time_in_sec', dest='training_time_in_sec', default=60, type=int, help='Training time in sec')

        parser.add_argument('-delays0', dest='delays0', default=1, type=int, help='delays type int')
        parser.add_argument('-delays1', dest='delays1', default=3, type=int, help='delays type int')
        parser.add_argument('-delays2', dest='delays2', default=1, type=int, help='delays type int')
        parser.add_argument('-shift0', dest='shift0', default=2, type=int, help='shift size type int')
        parser.add_argument('-shift1', dest='shift1', default=4, type=int, help='shift size type int')
        parser.add_argument('-shift2', dest='shift2', default=1, type=int, help='shift size type int')
        parser.add_argument('-mag0', dest='mag0', default=1.0, type=float, help='mag size type float')
        parser.add_argument('-mag1', dest='mag1', default=0.8, type=float, help='mag size type float')
        parser.add_argument('-mag2', dest='mag2', default=1.2, type=float, help='mag size type float')

        parser.add_argument('-delta_delays0', dest='delta_delays0', default=1, type=int, help='delays type int')
        parser.add_argument('-delta_delays1', dest='delta_delays1', default=1, type=int, help='delays type int')
        parser.add_argument('-delta_delays2', dest='delta_delays2', default=2, type=int, help='delays type int')
        parser.add_argument('-delta_shift0', dest='delta_shift0', default=4, type=int, help='shift size type int')
        parser.add_argument('-delta_shift1', dest='delta_shift1', default=1, type=int, help='shift size type int')
        parser.add_argument('-delta_shift2', dest='delta_shift2', default=1, type=int, help='shift size type int')
        parser.add_argument('-delta_mag0', dest='delta_mag0', default=3.0, type=float, help='mag size type float')
        parser.add_argument('-delta_mag1', dest='delta_mag1', default=3.0, type=float, help='mag size type float')
        parser.add_argument('-delta_mag2', dest='delta_mag2', default=2.0, type=float, help='mag size type float')

    def set_parameters(self, params):
        self.node = params.node
        self.density = params.density
        self.input_scale = params.input_scale
        self.rho = params.rho
        self.fb_scale = params.fb_scale
        self.leaking_rate = params.leaking_rate
        self.average_window = params.average_window
        self.no_class = params.no_class

        self.num_of_augmented_data = params.num_of_augmented_data
        self.num_of_output_classes = params.num_of_output_classes
        self.training_time_in_sec = params.training_time_in_sec

        self.delays0 = params.delays0
        self.shift0 = params.shift0
        self.mag0 = params.mag0
        self.delays1 = params.delays1
        self.shift1 = params.shift1
        self.mag1 = params.mag1
        self.delays2 = params.delays2
        self.shift2 = params.shift2
        self.mag2 = params.mag2

        self.delta_delays0 = params.delta_delays0
        self.delta_shift0 = params.delta_shift0
        self.delta_mag0 = params.delta_mag0
        self.delta_delays1 = params.delta_delays1
        self.delta_shift1 = params.delta_shift1
        self.delta_mag1 = params.delta_mag1
        self.delta_delays2 = params.delta_delays2
        self.delta_shift2 = params.delta_shift2
        self.delta_mag2 = params.delta_mag2



class DataAugmentation(app_base.DataAugmentation):
    def __init__(self, parameters):
        self.delays = parameters[0]
        self.shift = parameters[1]
        self.mag = parameters[2]
        self.delta_delays = parameters[3]
        self.delta_shift = parameters[4]
        self.delta_mag = parameters[5]
        self.pre_pulse = 0
        self.num_of_data = 0
        self.past_data = []

    def get_augmented_data(self, pulse):
        #print('data1',pulse)
        data = []

        input = float(pulse)
       
        # オリジナルデータ格納
        data.append(input)



        # オリジナルのディレイデータ格納
        if self.delays > 0:
            for delay in range(self.delays):
                delay_data = 0
                if self.shift == 0:
                    delay_data = input
                else:
                    if self.num_of_data >= (1 + delay) * self.shift:
                        delay_data = self.past_data[-(1 + delay) * self.shift][0] *  self.mag
                data.append(delay_data)

        # 階差データ格納
        delta = 0.0
        if self.num_of_data > 0:
            delta = float(input - self.past_data[-1][0])
        data.append(delta)
        

        # 階差のディレイデータ格納
        if self.delta_delays > 0:
            for delay in range(self.delta_delays):
                delay_data = 0
                if self.delta_shift == 0:
                    delay_data = input
                else:
                    if self.num_of_data >= (1 + delay) * self.delta_shift:
                        delay_data = self.past_data[-(1 + delay) * self.delta_shift][1] *  self.delta_mag
                data.append(delay_data)




        self.past_data.append((input,delta)) 
        self.num_of_data += 1

        return data


###############################################################################
# Train
###############################################################################
class Ball():
    def __init__(self, x, y, ax, direction):
        self.x = x
        self.y = y
        self.ax = ax
        self.r = 12
        self.interval = 0.2
        self.x_step = 0
        self.y_step = -40
        self.direction = direction
        if self.direction == 1:
            self.mark = '↑'
        elif self.direction == 2:
            self.mark = '↓'
        elif self.direction == 3:
            self.mark = '←'
        elif self.direction == 4:
            self.mark = '→'
        elif self.direction == 5:
            self.mark = 'B'
        elif self.direction == 6:
            self.mark = 'A'
        else:
            print('Direction Error')
            exit(1)
        self.anno = self.ax.annotate(self.mark, xy = (self.x-5, self.y-5), size = 12, color = "white")

    def move(self):
        self.x += self.x_step
        self.y += self.y_step
        self.circle = patches.Circle((self.x, self.y), self.r, fc='r', ec='r')
        self.anno.set_x(self.x-5)
        self.anno.set_y(self.y-5)
        self.ax.add_patch(self.circle)


    def clear(self):
        self.circle.remove()
        #self.anno.remove()

    def get_pos(self):
        return self.x, self.y

    def get_direction(self):
        return self.direction

class TrainingApp(app_base.TrainingApp):
    def __init__(self, parametes):
        super().__init__()
        self.label = 999
        self.parametes = parametes

        self.data0 = DataAugmentation([self.parametes.delays0, self.parametes.shift0, self.parametes.mag0, self.parametes.delta_delays0, self.parametes.delta_shift0, self.parametes.delta_mag0])
        self.data1 = DataAugmentation([self.parametes.delays1, self.parametes.shift1, self.parametes.mag1, self.parametes.delta_delays1, self.parametes.delta_shift1, self.parametes.delta_mag1])
        self.data2 = DataAugmentation([self.parametes.delays2, self.parametes.shift2, self.parametes.mag2, self.parametes.delta_delays2, self.parametes.delta_shift2, self.parametes.delta_mag2])


    def start(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_aspect('equal', adjustable='box')

        self.ax.set_xlim(-100, 200)
        self.ax.set_ylim(-150, 150)
        self.circles_destance = 50
        self.bpm = 100

        #random.seed(10)
        np.random.seed(10)
        self.draw_base_circles()
        self.wait_seconds(self.ax, 5)


        balls = []
        cnt = 0
        while True:
            self.interval = 60 / self.bpm

            x = 0
            y = 100
            r = 10
            direction = np.random.randint(0,7)

            if cnt % 2 == 0:
                # up
                if direction == 1:
                    x = self.top_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # down
                elif direction == 2:
                    x = self.buttom_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # left
                elif direction == 3:
                    x = self.left_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # right
                elif direction == 4:
                    x = self.rihgt_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # B button
                elif direction == 5:
                    x = self.B_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # A button
                elif direction == 6:
                    x = self.A_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                else:
                    x = 0
                    y = 0
            cnt += 1
            self.label = 0

            # 円 移動
            for ball in balls:
                ball.move()



            # 円描画
            plt.draw()

            # アノテーション決定
            for ball in balls:
                x, y = ball.get_pos()
                if y == -100:
                    self.label = ball.get_direction()
                    


            # 描画ポーズ
            plt.pause(self.interval)


            remained_balls = []
            for ball in balls:
                ball.clear()
                x, y = ball.get_pos()
                if self.is_on_mark(x,y) != True:
                    remained_balls.append(ball)
                else:
                    del ball

            balls = remained_balls

            if self.is_exit:
                print('stopped TrainingApp')
                plt.close(self.fig)
                return

    def stop(self):
        super().stop()

    def get_rawdata_and_labels(self, rawdata):
        data = rawdata , self.get_label()
        return data

    def get_data(self, csv_data):
        pulse00 = csv_data[0]
        pulse01 = csv_data[1]
        pulse02 = csv_data[2]
        pulse03 = csv_data[3]
        buttons = csv_data[4]
        label   = csv_data[5]
        return pulse00, pulse01, pulse02, label

    def prepare_data(self, csv_data):
        pulse00, pulse01, pulse02, label = self.get_data(csv_data)
        pulses = np.array(self.data0.get_augmented_data(pulse00) + self.data1.get_augmented_data(pulse01) + self.data2.get_augmented_data(pulse02))
        labels = np.array([float(int(label) == 1), float(int(label) == 2), float(int(label) == 3), float(int(label) == 4), float(int(label) == 5), float(int(label) == 6)]).reshape((1, -1))
        return pulses, labels

    def draw_base_circles(self):
        #fig = plt.figure()
        #board = self.ax(xlim=(-150, 150), ylim=(-150, 150))

        circles_destance_x = self.circles_destance
        r = 12
        init_x =   -75
        init_y =  -100
        #base_circle   = [0,                 0,                r]
        self.top_circle    = [init_x, init_y, r]
        init_x += 50
        self.buttom_circle = [init_x, init_y, r]
        init_x += 50
        self.left_circle   = [init_x, init_y, r]
        init_x += 50
        self.rihgt_circle  = [init_x, init_y, r]
        init_x += 50
        self.B_circle      = [init_x, init_y, r]
        init_x += 50
        self.A_circle      = [init_x, init_y, r]

        circles = [self.top_circle,self.buttom_circle,self.left_circle,self.rihgt_circle,self.B_circle,self.A_circle]

        for circle in circles:
            x, y, r = circle
            circle = patches.Circle((x, y), r, fc='w', ec='b')
            self.ax.add_patch(circle)

    def wait_seconds(self, ax, sec):
        for i in range(sec):
            remained_time = sec - i
            anno = ax.annotate(str(remained_time), xy = (-6, -6), size = 12, color = "red")
            plt.draw()
            plt.pause(1)
            anno.remove()



    def is_on_mark(self, x, y):
        #return abs(x) >= self.circles_destance or abs(y) >= self.circles_destance
        return y <= -100

    def get_label(self):
        return self.label

    def is_alive(self):
        return super().is_alive()


###############################################################################
# Predict
###############################################################################
import balloon
import pygame
import time

class PredictApp(app_base.PredictApp):
    def __init__(self, parametes):
        super().__init__()
        self.parametes = parametes
        self.data0 = DataAugmentation([self.parametes.delays0, self.parametes.shift0, self.parametes.mag0, self.parametes.delta_delays0, self.parametes.delta_shift0, self.parametes.delta_mag0])
        self.data1 = DataAugmentation([self.parametes.delays1, self.parametes.shift1, self.parametes.mag1, self.parametes.delta_delays1, self.parametes.delta_shift1, self.parametes.delta_mag1])
        self.data2 = DataAugmentation([self.parametes.delays2, self.parametes.shift2, self.parametes.mag2, self.parametes.delta_delays2, self.parametes.delta_shift2, self.parametes.delta_mag2])
        E_NEUTRAL = pygame.event.Event(pygame.USEREVENT, attr1='E_NEUTRAL')
        E_UP = pygame.event.Event(pygame.USEREVENT, attr1='E_UP')
        E_DOWN = pygame.event.Event(pygame.USEREVENT, attr1='E_DOWN')
        E_LEFT = pygame.event.Event(pygame.USEREVENT, attr1='E_LEFT')
        E_RIGHT = pygame.event.Event(pygame.USEREVENT, attr1='E_RIGHT')
        E_B = pygame.event.Event(pygame.USEREVENT, attr1='E_B')
        E_A = pygame.event.Event(pygame.USEREVENT, attr1='E_A')
        self.game = balloon.Game([E_NEUTRAL, E_UP, E_DOWN, E_LEFT, E_RIGHT, E_B, E_A])
        self.moving_avg_win_size = 25
        self.is_neutral = True
        self.detect_time = time.time()


    def start(self):
        super().start()
        self.game.run()

    def stop(self):
        super().stop()

    def get_data(self, rawdata):
        pulse00 = rawdata[0]
        pulse01 = rawdata[1]
        pulse02 = rawdata[2]
        pulse03 = rawdata[3]
        buttons = rawdata[4]
        return pulse00, pulse01, pulse02

    def prepare_data(self, rawdata):
        pulse00, pulse01, pulse02 = self.get_data(rawdata)
        pulses = np.array(self.data0.get_augmented_data(pulse00) + self.data1.get_augmented_data(pulse01) + self.data2.get_augmented_data(pulse02))
        return pulses

    def set_predict_result(self, predicted):
        super().set_predict_result(predicted)

        if len(predicted[-self.moving_avg_win_size:-1]) == 0:
            print('P skip02')
            return

        up, down, left, right, B, A = np.mean(predicted[-self.moving_avg_win_size:-1], axis=0)
        max_value = max([up, down, left, right, B, A])
        self.game.setPredicts([up, down, left, right, B, A])

        return

    def is_alive(self):
        return super().is_alive()

if __name__=="__main__":
    training_app = TrainingApp()

