# -*- coding: utf-8 -*-
import os
import datetime
import argparse
import copy

import batch_training_and_predict
import csv

import time
from multiprocessing import Pool

from tqdm import tqdm

#global APP
# .env ファイルをロードして環境変数へ反映
from dotenv import load_dotenv
load_dotenv()

# 環境変数を参照
APP_NAME = os.getenv('APPLICATION_NAME')
exec("import {}".format(APP_NAME) )

# Set application class
exec("APP = {}".format(APP_NAME) )

parser = argparse.ArgumentParser(description='Hyper parameter.')
parser.add_argument('-cpu', dest='cpu', type=int, default=15, help='num of cpu core')
parser.add_argument('-csv_file', dest='csv_file', type=str, help='target data', required=True)
parser.add_argument('-save_dir', dest='save_dir', default='output', type=str, help='Directory for output files')

# Set hyper parametes and custom parameters for the application
parameters = APP.Parameters(parser)
params = parser.parse_args()

now = datetime.datetime.now()
save_dir = os.path.join(params.save_dir, now.strftime('%Y%m%d_%H%M%S'))

# Make directory if not exist
os.makedirs(save_dir, exist_ok=True)

n_cpu = params.cpu
csv_file = params.csv_file

def write_result_csv(out_filename, results_data):
    labels = list(results_data[0].keys())

    try:
        with open(out_filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            for elem in results_data:
                writer.writerow(elem)
    except IOError:
        print("I/O error")



def wraper(parameters):
    acc = batch_training_and_predict.main(APP, APP_NAME, parameters, csv_file, save_dir, is_save_chart=False, is_show_chart=False, is_save_model=False)

    params_str, params_dict = parameters.get_title_from_params()
    print(str('{:.4f}'.format(acc)), params_str)
    return acc, params_str, params_dict

def main(parameters):
    p = Pool(n_cpu)

    parameters_list = []
    print('Preparing a gird search. Please wait... ')
    with tqdm() as pbar:
        while parameters.set_next_grid_search_params():
            parameters_list.append(copy.deepcopy(parameters))
            pbar.update(1)

    result_data = []
    results = p.map(wraper, parameters_list)
    print('Results of grid search for', APP_NAME)
    for result in sorted(results, reverse=True):
        acc, params_str, params_dict = result
        result_dict = {}
        result_dict.update({'acc':acc})
        result_dict.update(params_dict)
        result_data.append(result_dict)
        print(acc, params_str)


    result_csv_filename = os.path.join(save_dir, 'results_grid_search_' + APP_NAME + '.csv')
    write_result_csv(result_csv_filename, result_data)
    print('Save result csv to ',result_csv_filename)


if __name__ == '__main__':

    try:
        main(parameters)
    except KeyboardInterrupt:
        print('closeing...')
    exit()

