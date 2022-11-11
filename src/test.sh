mv .env .env.bk
echo APPLICATION_NAME=app_thumbup >> .env
python batch_training_and_predict.py -csv_file output/train_log_20221101_115348.csv
echo APPLICATION_NAME=app_thumbup_new >> .env
python grid_search.py -csv_file output/thumbup_nc_data.csv -cpu 4
mv .env.bk .env
