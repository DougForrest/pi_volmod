#!/usr/bin/env python3

import argparse
import collections
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import sounddevice as sd
import subprocess
import sys
import queue

import config as cfg


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def to_decibels(x, scale=32767):
    return(20 * np.log10(np.sqrt(np.mean((x * scale) ** 2))))


class AudioCallback:
    def __init__(self,
                 samplerate=48000,
                 interval=200,
                 decibels_target=60,
                 decibels_upper_limit=70,
                 decibels_lower_limit=20,
                 scale=32767,
                 graph=None):

        self.samplerate = samplerate
        self.interval = interval
        self.scale = scale
        self.current_indata = None
        self.rolling_data = np.array([])
        self.current_decibels = None
        self.decibels_lst_len_max = 50
        self.decibels_lst = collections.deque(maxlen=self.decibels_lst_len_max)
        self.decibels_target = decibels_target
        self.decibels_upper_limit = decibels_upper_limit
        self.decibels_lower_limit = decibels_lower_limit
        self.current_volume = 5
        self.volume_mapping = {1: 'up', -1: 'down'}
        self.volume_adjustment_max_retries = 50
        self.volume_adjustment_q_len = 12
        self.volume_adjustment_q = collections.deque(maxlen=self.volume_adjustment_q_len)
        self.volume_retries = 0
        self.last_retry_time = None
        self.last_volume_adjustment_attempt = datetime.now()
        self.last_volume_adjustment = 0
        self.current_volume_adjustment = 0
        self.total_frames = 0
        self.volume_milliseconds_delay = 100
        self.frames_interval = int((self.samplerate / 1000) * self.interval)
        self.state = 'running'
        self.graph = graph


    def __call__(self, indata, frames, time, status):
        """Entry point of the callback"""
        # if self.graph:
        #     self.graph(indata, frames, time, status)
        self.total_frames += frames
        self.current_indata = indata.copy().flatten()
        self.rolling_data = np.concatenate((self.rolling_data,
                                            self.current_indata), axis=0)

        self.current_decibels = to_decibels(self.rolling_data, scale=self.scale)

        if self.graph:
                self.graph(self.current_decibels, self.decibels_lst)

        if self.total_frames >= self.frames_interval:
            self.rolling_data = self.rolling_data[-self.frames_interval:]
            self.total_frames = 0
            self.current_decibels = to_decibels(self.rolling_data, scale=self.scale)


            self.decibels_lst.append(self.current_decibels)
            print(f"current decibels {self.current_decibels}")
            if self.state == 'running':
                self.evalute_volume_level()
            if self.state == 'paused':
                self.watch_db_levels_for_change()

    def check_rolling_db_level(self, last_n=5, std_threshold=1):
        """Check for long running differences in volume level"""
        if len(self.decibels_lst) < self.decibels_lst_len_max:
            return (None)

        decibels = np.array(self.decibels_lst)

        db_range = self.decibels_upper_limit - self.decibels_lower_limit
        if (decibels.mean() - self.decibels_target) > (0.5 * (self.decibels_upper_limit - self.decibels_target)):
            print('rolling db upper limit')
            self.current_volume_adjustment = -1

        if (self.decibels_target - decibels.mean()) > (0.5 * (self.decibels_target - self.decibels_lower_limit)):
            print('rolling db lower limit')
            self.current_volume_adjustment = 1

    def check_for_multiple_down_vol_adjustments(self, last_n=4, std_threshold=1):
        """Return volume to normal after many down volume adjustments"""
        print(f"check_for_multiple_down_vol_adjustments {self.volume_adjustment_q}")
        if len(self.volume_adjustment_q) != self.volume_adjustment_q_len:
            return (None)

        curr_db_mean, curr_db_std, rolling_db_mean, rolling_db_std = self.calc_rolling_decibels(last_n=last_n, std_threshold=std_threshold)

        if (sum(self.volume_adjustment_q) < 0) and (current_db_mean <= self.decibels_target):
            print('volume up in response to multiple volume down adjustments')
            print(f"{sum(self.volume_adjustment_q)} {self.volume_adjustment_q[-1]}")
            self.current_volume_adjustment = 1

    def calc_rolling_decibels(self, last_n=15, std_threshold=2):
        if len(self.decibels_lst) < last_n:
            return (None)

        decibels = np.array(self.decibels_lst)
        rolling_dbs = decibels[:len(decibels) - last_n]
        current_dbs = decibels[-last_n:]

        return([current_dbs.mean(), current_dbs.std(),
                rolling_dbs.mean(), rolling_dbs.std()])

    def watch_db_levels_for_change(self, last_n=15, std_threshold=2):

        curr_db_mean, curr_db_std, rolling_db_mean, rolling_db_std = self.calc_rolling_decibels(last_n=last_n, std_threshold=std_threshold)

        print(f"paused rolling_dbs mean {rolling_db_mean} std { rolling_db_std}")
        print(f"paused current_dbs mean {curr_db_mean} std { curr_db_std}")
        if abs(curr_db_mean - rolling_db_mean) > (std_threshold * rolling_db_std):
            print('running')
            self.state = 'running'
            self.decibels_lst = collections.deque(maxlen=self.decibels_lst_len_max)

    def evalute_volume_level(self):

        self.last_volume_adjustment = self.current_volume_adjustment
        self.current_volume_adjustment = 0

        if self.current_decibels and self.current_decibels > self.decibels_upper_limit:
            print('upper limit')
            self.current_volume_adjustment = -1

        if self.current_decibels and self.current_decibels < self.decibels_lower_limit:
            print('lower limit')
            self.current_volume_adjustment = 1

        if self.current_volume_adjustment == 0:
            # self.check_rolling_db_level()
            self.check_for_multiple_down_vol_adjustments()

        if self.current_volume_adjustment != 0:
            self.prepare_volume_adjustment(self.current_volume_adjustment)


    def calc_ms_decay(self, volume_retries):
        return (2 ** (volume_retries + 2) * 100)

    def adjust_volume(self, direction, delay=200):
        print("#" * 80)
        print(f'adjusting volume {self.volume_mapping[direction]}')
        self.current_volume += direction
        print(f"volume estimate {self.current_volume}")
        self.volume_adjustment_q.append(direction)

        if cfg.environment == "prod":
            subprocess.Popen(cfg.key_map[self.volume_mapping[direction]].split(' '))

    def check_volume_retry_limit(self, direction):
        print(f'check_volume_retry_limit')
        print(f'volume_retries {self.volume_retries} {self.volume_mapping[direction]}')
        if self.volume_retries > self.volume_adjustment_max_retries:
            print('volume self.volume_retries > self.volume_adjustment_max_retries')
            if self.volume_mapping[direction] == 'up':
                print('paused')
                self.volume_retries = 0
                self.state = 'paused'
                return (None)
            if self.volume_mapping[direction] == 'down':

                raise Exception ("Volume retry limit reached check IR device")

    def prepare_volume_adjustment(self, direction, delay=200):
        now = datetime.now()

        milliseconds_past = (now - self.last_volume_adjustment_attempt).total_seconds() * 1000
        print(f'milliseconds_past {milliseconds_past}')
        retry = False

        if direction == self.last_volume_adjustment:
            retry = True
            self.check_volume_retry_limit(direction)
            if self.volume_mapping[direction] == 'up':
                delay = self.calc_ms_decay(self.volume_retries)
                print(f'delay {delay}')


        if milliseconds_past >= delay:
            if retry:
                self.volume_retries += 1
            else:
                self.volume_retries = 0
            self.last_volume_adjustment_attempt = now
            self.adjust_volume(direction)


def parse_args(args):

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', '--show-devices',
                        action='store_true',
                        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.show_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        parents=[parser])
    parser.add_argument('channels', type=int,
                        default=[1], nargs='*', metavar='CHANNEL',
                        help='input channels to plot (default: the first)')
    parser.add_argument('-d', '--device', type=int_or_str,
                        help='input device (numeric ID or substring)')
    parser.add_argument('-w', '--window',
                        type=float, default=200, metavar='DURATION',
                        help='visible time slot (default: %(default)s ms)')
    parser.add_argument('-i', '--interval', type=float, default=300,
                        help='rolling window interval size (in milliseconds (default: %(default)s))')
    parser.add_argument('-b', '--blocksize',
                        type=int, help='block size (in samples)')
    parser.add_argument('-r', '--samplerate',
                        type=float, help='sampling rate of audio device')
    parser.add_argument('-u', '--decibels_upper_limit',
                        type=int, default=70, help='decibels upper limit (default: %(default)s)')
    parser.add_argument('-l', '--decibels_lower_limit',
                        type=int, default=35, help='decibels lower limit  (default: %(default)s)')
    parser.add_argument('-n', '--downsample',
                        type=int, default=10, metavar='N',
                        help='display every Nth sample (default: %(default)s)')
    parser.add_argument('-t', '--decibels_target',
                        type=int, default=60, metavar='N',
                        help='target decibels (default: %(default)s)')
    args = parser.parse_args(remaining)

    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')

    args.mapping = [c - 1 for c in args.channels]

    print(f"Running at target decibels {args.decibels_target}")
    print(f"upper limit decibels {args.decibels_upper_limit}")
    print(f"lower limit decibels {args.decibels_lower_limit}")
    # print(f'args.window {args.window}')
    # print(f'args.samplerate {args.samplerate}')
    # print(f'args.downsample {args.downsample}')
    # print(f'args.mapping {args.mapping}')
    # print(f"args.samplerate {args.samplerate}")
    print(f"args.device {args.device}")
    # print(f"args.channels {args.channels}")
    return(args)



class LinePlot:

    def __init__(self, window=None, samplerate=None,
                 downsample=None, interval=None,
                 mapping=None):
        self.window = window
        self.samplerate = samplerate
        self.downsample = downsample
        self.interval = interval
        self.mapping = mapping
        self.length = int(self.window * self.samplerate / (1000 * self.downsample))
        self.plotdata = np.zeros(self.length)
        self.q = queue.Queue()
        self.ani = None

        self.create_plot()

    def create_plot(self):

        fig, ax = plt.subplots()
        self.lines = ax.plot(self.plotdata)
        ax.axis((0, len(self.plotdata), 30, 80))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)
        self.ani = FuncAnimation(fig, self.update_plot, interval=self.interval, blit=True)

    def __call__(self, indata, data2):
        self.q.put([[indata.copy()], data2.copy()])
        # self.q.put(indata.copy().flatten()[::self.downsample])
        # self.q.put(indata[::self.downsample, self.mapping])
        # self.q.put(indata[::self.downsample, self.mapping])

    def update_plot(self, frame):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """

        while True:
            try:
                res = self.q.get_nowait()
            except queue.Empty:
                break
            print(res)
            data, data_q = res
            shift = len(data)
            # print(shift)
            print(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:] = data
            self.lines[0].set_ydata(np.array(data_q))
        # for column, line in enumerate(self.lines):
        # self.lines[0].set_ydata(self.plotdata[:])

        # self.lines[0].set_ydata(data2)
        return self.lines

    def update_plot2(self, frame):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """

        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines


def main(args):

    args = parse_args(args)

    lineplt = LinePlot(window=args.window,
                       samplerate=args.samplerate,
                       downsample=args.downsample,
                       interval=30,
                       mapping =args.mapping)

    callback_obj = AudioCallback(decibels_target=args.decibels_target,
                                 decibels_upper_limit=args.decibels_upper_limit,
                                 decibels_lower_limit=args.decibels_lower_limit,
                                 interval=args.interval,
                                 samplerate=args.samplerate,
                                 graph=lineplt)

    try:

        with sd.InputStream(device=args.device, channels=max(args.channels),
                            samplerate=args.samplerate, callback=callback_obj):
            plt.show()
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()

    except KeyboardInterrupt:
        quit('Interrupted by user')

    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))


if __name__ == '__main__':
    main(sys.argv[1:])
