# -*- coding: utf-8 -*-
import data_streamer_base


class DataStreamer(data_streamer_base.DataStreamerBase):
    def __init__(self, port=None):
        super().__init__()
        self.is_exit = False

    def run(self):
        while True:
            if self.callback is not None:
                self.callback(self.data)
            if self.is_exit:
                continue

    def set_callback(self, callback):
        super().set_callback(callback)
        self.is_exit = False

    def stop(self):
        self.is_exit = True


def callback(arg):
    print('callback',arg)

if __name__=="__main__":
    com = DataStreamer(port='COM5')
    com.set_callback(callback)
