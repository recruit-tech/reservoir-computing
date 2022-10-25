# -*- coding: utf-8 -*-
import data_streamer_base
import serial



class DataStreamer(data_streamer_base.DataStreamerBase):
    def __init__(self, port):
        self.is_exit = False
        self.port = port
        super().__init__()

    def run(self):
        print('run')
        self.serial = serial.Serial(self.port,115200,timeout=100)
        self.wait_until_first_data()

        while True:
            res = self.serial.readline()
            self.data = res.decode().replace( '\r\n' , '' ).split(',')
            if self.callback is not None:
                self.callback(self.data)

            #print('self.is_exit',self.is_exit)
            if self.is_exit:
                self.serial.close()
                print('stopped DataStreamer')
                return

    def set_callback(self, callback):
        super().set_callback(callback)
        self.is_exit = False


    #def end(self):
    #    #self.serial.close()
    #    super().end()

    #def stop(self):
    #    self.is_exit = True

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

def callback(arg):
    print('callback',arg)

if __name__=="__main__":
    serial = DataStreamer('COM5')
    serial.set_callback(callback)
