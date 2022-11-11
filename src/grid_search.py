# -*- coding: utf-8 -*-
import os
import datetime
import argparse
import copy

import batch_training_and_predict

import time
from multiprocessing import Pool

from tqdm import tqdm

def wraper(parameters):
    #print(parameters.get_title_from_params())
    acc = batch_training_and_predict.main(APP, APP_NAME, parameters, params.csv_file, save_dir, is_show_chart=False)
    print(acc, parameters.get_title_from_params())
    return acc, parameters.get_title_from_params()

def main(app, app_name, parameters, csv_file, save_dir):
    parameters_list = []
    print('Preparing a gird search. Please wait... ')
    with tqdm() as pbar:
        while parameters.set_next_grid_search_params():
            parameters_list.append(copy.deepcopy(parameters))
            pbar.update(1)

    p = Pool(n_cpu) 
    accs = p.map(wraper, parameters_list)
    for result in sorted(accs, reverse=True):
        print(result)
    #for params in parameters_list:
    #    acc = batch_training_and_predict.main(app, app_name, params, csv_file, save_dir, is_show_chart=False)
    #    print('acc', acc, params.get_title_from_params())

    #while parameters.set_next_grid_search_params():
    #    acc = batch_training_and_predict.main(app, app_name, parameters, csv_file, save_dir, is_show_chart=False)
    #    print('acc', acc, parameters.get_title_from_params())

if __name__ == '__main__':
    # .env ファイルをロードして環境変数へ反映
    from dotenv import load_dotenv
    load_dotenv()

    # 環境変数を参照
    APP_NAME = os.getenv('APPLICATION_NAME')
    exec("import {}".format(APP_NAME) )

    # Set application class
    exec("APP = {}".format(APP_NAME) )
    print('Application',APP)

    parser = argparse.ArgumentParser(description='Hyper parameter.')
    parser.add_argument('-cpu', dest='cpu', type=int, default=15, help='num of cpu core')
    parser.add_argument('-csv_file', dest='csv_file', type=str, help='target data', required=True)
    parser.add_argument('-save_dir', dest='save_dir', default='output', type=str, help='Directory for output files')

    # Set hyper parametes and custom parameters for the application
    parameters = APP.Parameters(parser)
    params = parser.parse_args()
    print('params',params)

    # Set parametes as global variable
    #csv_file = params.csv_file
    #save_dir = params.save_dir

    now = datetime.datetime.now()
    save_dir = os.path.join(params.save_dir, now.strftime('%Y%m%d_%H%M%S'))

    # Make directory if not exist
    os.makedirs(save_dir, exist_ok=True)

    n_cpu = params.cpu

    try:
        main(APP, APP_NAME, parameters, params.csv_file, save_dir)
    except KeyboardInterrupt:
        print('closeing...')
    exit()

