# -*- coding: utf-8 -*-
import data_streamer_base
import serial



class DataStreamer(data_streamer_base.DataStreamerBase):
    def __init__(self, port):
        super().__init__()
        self.com = serial.Serial(port,115200,timeout=100)
        self.wait_until_first_data()
        self.is_exit = False

    def run(self):
        while True:
            res = self.com.readline()
            self.data = res.decode().replace( '\r\n' , '' ).split(',')
            if self.callback is not None:
                self.callback(self.data)
            if self.is_exit:
                #self.com.close()
                #print('stopped DataStreamer4Com')
                continue

    def set_callback(self, callback):
        super().set_callback(callback)
        self.is_exit = False

    def stop(self):
        self.is_exit = True

    def wait_until_first_data(self):
        while True:
            res = self.com.readline()
            n = int.from_bytes( res, 'little' )

            try:
                data = res.decode().split(',')
            except UnicodeDecodeError as e:
                print('e',e)
                continue
            
            if len(data):
                print('data',data)
                break

def callback(arg):
    print('callback',arg)

if __name__=="__main__":
    com = DataStreamer('COM5')
    com.set_callback(callback)
