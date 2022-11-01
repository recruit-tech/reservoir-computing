# -*- coding: utf-8 -*-
import data_streamer_base

class DataStreamer(data_streamer_base.DataStreamerBase):
    def __init__(self, port=None):
        super().__init__()

    def run(self):
        self.is_exit = False

        while True:
            if self.callback is not None:
                self.callback(self.data)

            if self.is_exit:
                print('stopped DataStreamer')
                return


def callback(arg):
    print('callback',arg)

if __name__=="__main__":
    com = DataStreamer(port='COM5')
    com.set_callback(callback)
