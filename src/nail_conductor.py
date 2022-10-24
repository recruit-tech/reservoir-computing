# -*- coding: utf-8 -*-
import numpy as np
import time
import datetime
import csv
import argparse

from model import ESN, Tikhonov

import os
import pickle

parser = argparse.ArgumentParser(description='Hyper parameter.')

parser.add_argument('-in_model_filename', dest='in_model_filename', default=None, type=str, help='Input model file (*.pkl). Skip training when model is set.')
parser.add_argument('-save_dir', dest='save_dir', default='output', type=str, help='Directory for output files')


def csv_writer(file_name, data):
    f = open(file_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()


class ScalingShift:
    def __init__(self, scale, shift):
        '''
        :param scale: 出力層のスケーリング（scale[n]が第n成分のスケーリング）
        :param shift: 出力層のシフト（shift[n]が第n成分のシフト）
        '''
        self.scale = np.diag(scale)
        self.shift = np.array(shift)
        self.inv_scale = np.linalg.inv(self.scale)
        self.inv_shift = -np.dot(self.inv_scale, self.shift)

    def __call__(self, x):
        return np.dot(self.scale, x) + self.shift

    def inverse(self, x):
        return np.dot(self.inv_scale, x) + self.inv_shift



class Training():
    def __init__(self, data_streamer, model, app, optimizer, out_csv_filename, out_model_filename, measurement_time = 60):
        self.data_streamer = data_streamer
        self.data_streamer.set_callback(self.train)
        self.model = model
        self.app = app
        self.measurement_time = measurement_time
        self.optimizer = optimizer
        self.out_csv_filename = out_csv_filename
        self.out_model_filename = out_model_filename
        self.save_data = []
        self.model.init_train_online()

        self.start_time = time.time()
        self.data_streamer.start()
        self.app.start()

    def train(self, data):
        now_time = time.time()
        if now_time - self.start_time >= self.measurement_time:
            self.data_streamer.stop()
            self.app.stop()
            csv_writer(self.out_csv_filename, self.save_data)
            self.model.finish_train_online(self.optimizer)
            pickle.dump(self.model, open(self.out_model_filename, 'wb'))

            return

        print('training',[self.app.get_data(data)])
        self.save_data.append(self.app.get_data(data))
        pulses, labels = self.app.prepare_data(data)
        self.model.train_online(pulses, labels, self.optimizer) 



class Predict():
    def __init__(self, data_streamer, model, app, measurement_time = 0):
        self.data_streamer = data_streamer
        self.data_streamer.set_callback(self.predict)
        self.model = model
        self.app = app
        self.measurement_time = measurement_time
        self.model.init_predict_online()

        self.start_time = time.time()
        self.app.start()

    def predict(self, data):
        now_time = time.time()
        if now_time - self.start_time >= self.measurement_time and self.measurement_time != 0:
            self.data_streamer.stop()
            self.app.stop()
            return

        print('predict',[self.app.get_data(data)])
        pulses = self.app.prepare_data(data)
        predict_result = self.model.predict_online(pulses) 
        self.app.set_predict_result(predict_result)

        return

def main(ds, app):

    # Set hyper parametes and custom parameters for the application
    parametes = app.Parameters(parser)


    # Create or load the model
    if in_model_filename is None:

        # 出力のスケーリング関数
        output_func = ScalingShift([1.0], [1.0])

        # Create model with the hyper parameters
        model = ESN(parametes.num_of_input_data, 
                    parametes.num_of_output_classes, 
                    parametes.node, 
                    density=parametes.density,
                    input_scale=parametes.input_scale,
                    rho=parametes.rho,
                    fb_scale=parametes.fb_scale,
                    leaking_rate=parametes.leaking_rate,
                    classification = parametes.no_class, 
                    average_window=parametes.average_window)
    else:

        # Load a model file
        with open(in_model_filename, 'rb') as web:
            model = pickle.load(web)

    training_app = app.TrainingApp(parametes)
    optimizer = Tikhonov(parametes.node, parametes.num_of_output_classes, 0.1)

    Training(ds, model, training_app, optimizer, out_csv_filename, out_model_filename, measurement_time = 10)
    print('Training done')
    predict_app = app.PredictApp(parametes)
    Predict(ds, model, predict_app)

    predict_app.start()



import data_streamer_serial
import app_famicom
import app_thumbup

if __name__=="__main__":
    params = parser.parse_args()

    # Set parametes as global variable
    in_model_filename = params.in_model_filename
    save_dir = params.save_dir

    # Make directory if not exist
    os.makedirs(save_dir, exist_ok=True)

    # make csv and model file names
    now = datetime.datetime.now()
    out_csv_filename = os.path.join(save_dir, 'train_log_' + now.strftime('%Y%m%d_%H%M%S') + '.csv')
    out_model_filename = os.path.join(save_dir, 'model_' + now.strftime('%Y%m%d_%H%M%S') + '.pkl')

    # Set com port as data streamer
    ds = data_streamer_serial.DataStreamer('COM5')

    # Set application class
    #app = app_famicom
    app = app_thumbup

    main(ds, app)
