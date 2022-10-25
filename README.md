# reservoir-computing

## Real time training and predict framework
This program is a framework of reservoir-computing that adopts multiple applications.
It runs on basically PC(Ubuntu).

You might need to implement codes on a device like arduino for getting data from sensors. 

Here is a sample code which runs on arduino for 4 sensors and 3 buttons. You can custom it for your applications.

./src/arduino/puls_sensors_sample/puls_sensors_sample.ino


![Overview diagram](imgs/overview.png "overview")

### Usage

how to run a thumup detection with reservoir-computing
 ```
git clone https://github.com/recruit-tech/reservoir-computing.git
cd reservoir-computing/src
python nail_conductor.py
 ```

To quit the program, push [q] key on the screen or ctrl + c on the terminal.

You can also run a cardboard controller( famicom ) when you change code as follows in ./src/nail_conductor.py

 ```
    # Set application class
    app = app_famicom
    #app = app_thumbup
 ```

You will see the following images. (upper:training screen, lower: predict screen)
![famicom training](imgs/famicom_training.png "famicom_training")
![famicom predict](imgs/famicom_predict.png "famicom_predict")


### How to make YOUR APPs.
If you would like to make YOUR APPs, you only need to write 4 claasses which is in the orange frame on folliwing diagram.

![Class diagram](imgs/class_diagram.png "class_diagram")



