# -*- coding: utf-8 -*-
import data_streamer_base
import serial as ser



class DataStreamer(data_streamer_base.DataStreamerBase):
    def __init__(self, port):
        self.port = port
        self.serial = ser.Serial(self.port,115200,timeout=100)
        super().__init__()

    def run(self):
        self.is_exit = False
        self.wait_until_first_data()

        while True:
            res = self.serial.readline()
            self.data = res.decode().replace( '\r\n' , '' ).split(',')
            if self.callback is not None:
                self.callback(self.data)

            if self.is_exit:
                #self.serial.close()
                print('stopped DataStreamer')
                return

    def wait_until_first_data(self):
        while True:
            res = self.serial.readline()
            n = int.from_bytes( res, 'little' )

            try:
                data = res.decode().split(',')
            except UnicodeDecodeError as e:
                print('e',e)
                continue
            
            if len(data):
                print('data',data)
                break

def callback1(arg):
    a = 0
    print('callback1',arg)

def callback2(arg):
    a = 0
    print('callback2',arg)

import time 
if __name__=="__main__":
    print('Start')
    ds = DataStreamer('COM3')
    ds.begin(callback1)
    time.sleep(5)

    print('Change callback')
    ds.begin(callback2)
    time.sleep(5)

    print('Suspend')
    ds.end()
    time.sleep(5)

    ds.begin(callback1)
    print('Resume')
    time.sleep(5)

    ds.end()
    ds.kill()
