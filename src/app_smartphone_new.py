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
        self._idx = 0
        self._num_of_grid_search = 0

        
    def add_hyper_parameters(self, parser):
        # Hyper parametes for reserver computing 
        parser.add_argument('-node', dest='node', default=700, type=int, help='number of node')
        parser.add_argument('-density', dest='density', default=0.4, type=float, help='density')
        parser.add_argument('-input_scale', dest='input_scale', default=0.005, type=float, help='input scale')
        parser.add_argument('-rho', dest='rho', default=1.0, type=float, help='rho')
        parser.add_argument('-fb_scale', dest='fb_scale', default=None, type=float, help='fb scale')
        parser.add_argument('-leaking_rate', dest='leaking_rate', default=0.05, type=float, help='leaking rate')
        parser.add_argument('-average_window', dest='average_window', default=1, type=int, help='average window size')
        parser.add_argument('--no_classification', dest='no_class', action='store_false', help='no class')


    def add_custome_perametes(self, parser):
        parser.add_argument('-num_of_input_data', dest='num_of_input_data', default=6, type=int, help='num of input data (num of sensors)')
        parser.add_argument('-num_of_augmented_data', dest='num_of_augmented_data', default=6, type=int, help='num of input data (num of sensors)')
        parser.add_argument('-num_of_output_classes', dest='num_of_output_classes', default=4, type=int, help='num of output claasses')
        parser.add_argument('-training_time_in_sec', dest='training_time_in_sec', default=129, type=int, help='Training time in sec')
        parser.add_argument('--delta', dest='delta', default=0, type=int,                    help='if 0 no-delta else add 1 delta data with [value] times ')
        parser.add_argument('--train_random', dest='train_random', action='store_true',                    help='this option is used random train data.')
        parser.add_argument('--moving_average', dest='moving_average',                    default=5, type=int, help='moving average')


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

        self.delta = params.delta
        self.train_random = params.train_random
        self.moving_average = params.moving_average


    def set_next_grid_search_params(self):
        if self._idx >= self._num_of_grid_search and self._idx != 0:
            return False

        if self._idx == 0:
            self._grid_search_params_list = []
            for node in [800]:
                for density in [0.4, 0.5, 0.6]:
                    for input_scale in [0.001, 0.002, 0.005, 0.008]:
                        for rho in [1.05,1.1,1.15,1.2,1.4]:
                            for fb_scale in [None,]:
                                for leaking_rate in [0.15, 0.2, 0.25]:
                                    for average_window in [1,2,5]:
            #for node in [800]:
            #    for density in [0.4, 0.6, 0.8]:
            #        for input_scale in [0.005, 0.01, 0.05]:
            #            for rho in [0.4,0.8,0.9,1.0,1.1]:
            #                for fb_scale in [None,]:
            #                    for leaking_rate in [0.1, 0.2, 0.4, 0.8]:
            #                        for average_window in [1,5,10,30]:
            #for node in [900,800,700]:
            #    for density in [0.1,0.2,0.4]:
            #        for input_scale in [0.001,0.002]:
            #            for rho in [1.30,1.325,1.350,1.375,1.4]:
            #                for fb_scale in [None,]:
            #                    for leaking_rate in [0.6,0.8]:
            #                        for average_window in [1,]:
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

        param_str = ['n' + f'{self.node:04}', \
                     'd' + str('{:.3f}'.format(self.density)), \
                     'i' + str('{:.3f}'.format(self.input_scale)), \
                     'r' + str('{:.3f}'.format(self.rho)), \
                     'f' + str('{}'.format(self.fb_scale)), \
                     'l' + str('{:.3f}'.format(self.leaking_rate)), \
                     'w' + f'{self.average_window:03}' ]

        param_dict = {'node' : self.node, \
                      'density' : self.density, \
                      'input_scale' : self.input_scale, \
                      'rho' : self.rho, \
                      'fb_scale' : self.fb_scale, \
                      'leaking_rate' : self.leaking_rate, \
                      'average_window' : self.average_window}

        return "-".join(param_str), param_dict

class DataAugmentation(app_base.DataAugmentation):
    def __init__(self, parameters=[]):
        self.parameters = parameters
        self.delta = self.parameters[0]
        #self.delays = delays
        #self.skip = skip
        self.pre_pulse = 0

    def get_augmented_data(self, pulse):
        data = []
        #input = float(pulse)

        #input = [float(p) for p in pulse]
        #print('type:', type(input))
        # data.append(input)

        for pulse_data in pulse:
            data.append(float(pulse_data))

        if self.delta > 0:
            delta = float(input - self.pre_pulse)
            # data.append(delta)
            x_delta = float(delta * self.delta)
            data.append(x_delta)
        #self.pre_pulse = input
        # print('data',data)
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
            self.mark = 'in'
        elif self.direction == 2:
            self.mark = 'out'
        elif self.direction == 3:
            self.mark = '???'  # ????????????
        elif self.direction == 4:
            self.mark = '???'  # ????????????
        else:
            print('Direction Error')
            exit(1)
        self.anno = self.ax.annotate(self.mark, xy=(
            self.x, self.y), size=12, color='white', fontweight='bold', ha='center')

    def move(self):
        self.x += self.x_step
        self.y += self.y_step
        self.circle = patches.Circle((self.x, self.y), self.r, fc='r', ec='r')
        self.anno.set_x(self.x-2)
        self.anno.set_y(self.y-5)
        self.ax.add_patch(self.circle)
        # self.ax.add_patch(self.anno)
        # plt.draw()
        # if self.is_on_mark(x,y):
        #    self.label = direction

        # plt.pause(self.interval)
        # self.circle.remove()
        #self.label = 0

        # if self.is_on_mark(x,y):
        #    break

    def clear(self):
        self.circle.remove()
        # self.anno.remove()
        self.anno.set_y(self.y-50)

    def get_pos(self):
        return self.x, self.y

    def get_direction(self):
        return self.direction


class TrainingApp(app_base.TrainingApp):
    def __init__(self, parametes):
        super().__init__()
        self.parametes = parametes
        # threading.Thread.__init__(self)
        #self.stop = False
        self.data = DataAugmentation([self.parametes.delta,])
        # self.run()
        #matplotlib.use('TkAgg')


    def draw_base_circles(self):
        #fig = plt.figure()
        #board = self.ax(xlim=(-150, 150), ylim=(-150, 150))

        circles_destance_x = self.circles_destance
        r = 12
        init_x = -50
        init_y = -100
        self.p_in_curcle = [init_x, init_y, r]
        self.p_out_circle = [init_x, init_y, r]
        init_x += 100
        self.s_left_circle = [init_x, init_y, r]
        self.s_right_circle = [init_x, init_y, r]

        circles = [self.s_left_circle, self.s_right_circle,
                   self.p_in_curcle, self.p_out_circle]

        for circle in circles:
            x, y, r = circle
            circle = patches.Circle((x, y), r, fc='w', ec='b', lw='3')
            self.ax.add_patch(circle)

    def wait_seconds(self, ax, sec):
        for i in range(sec):
            remained_time = sec - i
            anno = ax.annotate(str(remained_time),
                               xy=(-6, -6), size=15, color='red')
            plt.draw()
            plt.pause(1)
            anno.remove()

    def getTrainData(self):
        index = 0
        train_data_pattern = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                              0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        dataNum = len(train_data_pattern)

        while True:
            if index >= dataNum:
                index = 0
            # yield index

            yield train_data_pattern[index]
            index += 1

    def start(self):
        super().start()
        loop = True
        self.label = 999
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_aspect('equal', adjustable='box')
        #plt.get_current_fig_manager().window.wm_geometry("+1000+200")

        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 150)
        self.circles_destance = 50
        self.bpm = 70

        np.random.seed(2)
        self.draw_base_circles()
        self.wait_seconds(self.ax, 5)

        #global label
        #global is_exit

        balls = []
        cnt = 0

        #if not self.parametes.train_random:
        getData = self.getTrainData()

        while loop:
            self.interval = 60 / self.bpm

            x = 0
            y = 100
            r = 10
            # direction = random.randint(0, N_y)
            # direction = np.random.randint(0, N_y + 1)
            # direction = getTrainData()
            # print('direction:', direction)

            # if True:
            if cnt % 2 == 0:
                #if self.parametes.train_random:
                #    direction = np.random.randint(0, N_y + 1)
                #else:
                direction = next(getData)
                # pinch_in
                if direction == 1:
                    x = self.p_in_curcle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # pinch_out
                elif direction == 2:
                    x = self.p_out_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # swipe_left
                elif direction == 3:
                    x = self.s_left_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                # swipe_right
                elif direction == 4:
                    x = self.s_right_circle[0]
                    y = 100
                    balls.append(Ball(x, y, self.ax, direction))

                else:
                    x = 0
                    y = 0
            cnt += 1
            self.label = 0

            # ??? ??????
            for ball in balls:
                ball.move()

            # ?????????
            plt.draw()

            # ???????????????????????????
            for ball in balls:
                x, y = ball.get_pos()
                if y == -100:
                    self.label = ball.get_direction()

            # ???????????????
            plt.pause(self.interval)

            remained_balls = []
            for ball in balls:
                ball.clear()
                x, y = ball.get_pos()
                if self.is_on_mark(x, y) != True:
                    remained_balls.append(ball)
                else:
                    del ball

            balls = remained_balls

            if self.is_exit:
                plt.close(self.fig)
                self.is_exit = False
                return

    def is_on_mark(self, x, y):
        return y <= -100

    def get_label(self):
        labels = self.label
        return labels

    def stop(self):
        super().stop()

    def get_rawdata_and_labels(self, rawdata):
        csv_data = rawdata + [self.get_label(),]
        return csv_data

    def get_data(self, csv_data):
        pulses = []
        pulses.append(csv_data[0])
        pulses.append(csv_data[1])
        pulses.append(csv_data[2])
        pulses.append(csv_data[3])
        pulses.append(csv_data[4])
        pulses.append(csv_data[5])
        label   = csv_data[6]
        return pulses, label

    def prepare_data(self, csv_data):
        pulses, label  = self.get_data(csv_data)
        pulses = self.data.get_augmented_data(pulses)
        labels = [float(int(label) == 1), float(int(label) == 2), float(int(label) == 3), float(int(label) == 4)]
        return pulses, labels

    def is_alive(self):
        return super().is_alive()

###############################################################################
# Predict
###############################################################################
import time
from websocket_server import WebsocketServer

class WsServer():
    def __init__(self, host, port):
        self.server = WebsocketServer(port=port, host=host)
        self.rcvData = []

    def newClient(self, client, server):
        print("Connected client : ", client['address'])
        self.server.send_message(client, "OK! Connected")

    def clientLeft(self, client, server):
        print("Disconnected : ", client['address'])

    def messageReceived(self, client, server, message):
        self.server.send_message(client, "OK, Received : " + message)
        self.rcvData.append(message)

    def sendMsgAllClient(self, message):
        print('sendMsgAllClient:', message)
        self.server.send_message_to_all(message)

    def runServer(self):
        self.server.set_fn_new_client(self.newClient)    # Client?????????
        self.server.set_fn_client_left(self.clientLeft)    # Client?????????
        self.server.set_fn_message_received(
            self.messageReceived)     # Client??????????????????
        self.server.run_forever()

import threading
class PredictApp(app_base.PredictApp):
    def __init__(self, parametes):
        super().__init__()
        self.parametes = parametes
        self.delta = self.parametes.delta
        self.data = DataAugmentation([self.delta,])
        HOST = 'localhost'  # IP????????????open
        PORT = 9999  # ??????????????????
        self.wsServer = WsServer(HOST, PORT)

        # Start receiving data from client
        wsThread = threading.Thread(target=self.wsStart)
        wsThread.setDaemon(True)    # ???????????????????????????????????????
        wsThread.start()
        print("Threading Start")

        #self.wsServer.runServer()

        # Start receiving data from client
        #self.wsThread = self.threading.Thread(target=wsStart)
        #self.wsThread.setDaemon(True)    # ???????????????????????????????????????
        #self.wsThread.start()
        #print("Threading Start")

        self.data = DataAugmentation([self.delta,])
        self.interval = 0.6
        self.classes = ['p_in', 'p_out', 's_left', 's_right']
        self.start_time = time.time() - self.interval * 2
        self.is_neutral = True

        self.wait = False

    def wsStart(self):
        self.wsServer.runServer()

    def start(self):
        super().start()
        loop = True
        self.now_time = time.time()
        self.wsServer.runServer()
        #self.model.init_predict_online()
        try:

            while loop:
                #self.now_time = time.time()
                if self.is_exit == True:
                    # thread.exit_loop()
                    self.is_exit = True
                    return

        except KeyboardInterrupt:
            self.is_exit = True
            return
            # sys.exit
        except Exception as e:
            self.is_exit = True
            print(e)
            return


    def stop(self):
        super().stop()

    def get_data(self, csv_data):
        pulses = []
        pulses.append(csv_data[0])
        pulses.append(csv_data[1])
        pulses.append(csv_data[2])
        pulses.append(csv_data[3])
        pulses.append(csv_data[4])
        pulses.append(csv_data[5])
        return pulses

    def prepare_data(self, rawdata):
        pulses = self.get_data(rawdata)
        pulses = np.array(self.data.get_augmented_data(pulses))
        return pulses

    def set_predict_result(self, predicted):
        super().set_predict_result(predicted)

        if len(predicted[-self.parametes.moving_average:-1]) == 0:
            print('P skip02')
            return

        p_in, p_out, s_left, s_right = np.mean(
            predicted[-self.parametes.moving_average:-1], axis=0)
        self.now_time = time.time()

        # ???????????????????????????????????????
        #print('self.now_time - self.start_time > self.interval and self.is_neutral',self.now_time,self.start_time,self.interval , self.is_neutral)
        if self.now_time - self.start_time > self.interval:# and self.is_neutral:
            max_value = max([p_in, p_out, s_left, s_right])
            print('max_value', max_value)
            if max_value > 0.5:
                max_index = [p_in, p_out, s_left,s_right].index(max_value)
                print(max_value, self.classes[max_index])
                self.wsServer.sendMsgAllClient(self.classes[max_index])
                self.wait = True
            #    self.is_neutral = False
            #else:
            #    self.is_neutral = True

        if self.wait:
            print("wait")
            self.start_time = time.time()
            self.wait = False


        return

    def is_alive(self):
        return super().is_alive()

if __name__=="__main__":
    training_app = TrainingApp()

