#!/usr/bin/env python3

import argparse
import collections
from datetime import datetime
import numpy as np
import os
import sounddevice as sd
import subprocess
import sys

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
                 decibels_lower_limit=40,
                 scale=32767):

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
        self.volume_adjustment_max_retries = 3
        self.volume_adjustment_lst_len = 4
        self.volume_adjustment_lst = collections.deque(maxlen=self.volume_adjustment_lst_len)
        self.volume_retries = 0
        self.last_retry_time = None
        self.last_volume_adjustment = 0
        self.current_volume_adjustment = 0
        self.total_frames = 0
        self.volume_milliseconds_delay = 100
        self.frames_interval = int((self.samplerate / 1000) * self.interval)
        self.state = 'running'

    def __call__(self, indata, frames, time, status):
        self.total_frames += frames
        self.current_indata = indata.copy().flatten()
        self.rolling_data = np.concatenate((self.rolling_data,
                                            self.current_indata), axis=0)

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

    def check_rolling_db_level(self, last_n=5, std_threshold=2):
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

    def check_for_multiple_down_vol_adjustments(self):
        """Return volume to normal after many down volume adjustments"""
        if len(self.volume_adjustment_lst) < self.volume_adjustment_lst_len:
            return (None)

        if (sum(self.volume_adjustment_lst) <= -(self.volume_adjustment_lst_len - 1)) and (self.volume_adjustment_lst[-1] == 0):
            print('volume up in response to multiple volume down adjustments')
            self.current_volume_adjustment = 1

    def watch_db_levels_for_change(self, last_n=15, std_threshold=2):

        if len(self.decibels_lst) < last_n:
            return (None)

        decibels = np.array(self.decibels_lst)
        rolling_dbs = decibels[:len(decibels) - last_n]
        current_dbs = decibels[-last_n:]
        print(f"paused rolling_dbs mean {rolling_dbs.mean()} std { rolling_dbs.std()}")
        print(f"paused current_dbs mean {current_dbs.mean()} std { current_dbs.std()}")
        if abs(current_dbs.mean() - rolling_dbs.mean()) > (std_threshold * rolling_dbs.std()):
            print('running')
            self.state = 'running'
            self.decibels_lst = collections.deque(maxlen=self.decibels_lst_len_max)
            # self.adjust_volume(-1)

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
            self.check_rolling_db_level()
            self.check_for_multiple_down_vol_adjustments()

        if self.current_volume_adjustment != 0:
            self.prepare_volume_adjustment(self.current_volume_adjustment)


    def calc_ms_decay(self, x):
        return (2 ** (x + 2) * 100)

    def retry_with_decay(self, direction):
        print('retry with decay')
        if self.volume_retries > self.volume_adjustment_max_retries:
            print('paused')
            self.volume_retries = 0
            self.state = 'paused'
            return (None)

        now = datetime.now()

        if not self.last_retry_time:
            self.last_retry_time = now
            self.volume_retries += 1

        milliseconds_past = (now - self.last_retry_time).total_seconds() * 1000
        print(f'milliseconds_past {milliseconds_past}')

        if milliseconds_past >= self.volume_milliseconds_delay:
            self.volume_retries += 1
            self.last_retry_time = now
            self.volume_milliseconds_delay = self.calc_ms_decay(self.volume_retries)
            print(f"self.volume_retries {self.volume_retries}")
            print(f"self.volume_milliseconds_delay {self.volume_milliseconds_delay}")
            self.adjust_volume(direction)

    def adjust_volume(self, direction):
        print(f'adjusting volume {self.volume_mapping[direction]}')

        self.current_volume += direction
        print(f"volume estimate {self.current_volume}")

        if cfg.environment == "prod":
            subprocess.Popen(cfg.key_map[self.volume_mapping[direction]].split(' '))

    def prepare_volume_adjustment(self, direction):

        if direction == self.last_volume_adjustment:
            if self.volume_mapping[direction] == 'up':
                self.retry_with_decay(direction)
                return (None)
            # else:
            #     self.volume_mapping[direction] == 'up':

        self.volume_retries = 0
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
    print(f'args.window {args.window}')
    print(f'args.samplerate {args.samplerate}')
    print(f'args.downsample {args.downsample}')
    print(f'args.mapping {args.mapping}')
    print(f"args.samplerate {args.samplerate}")
    print(f"args.device {args.device}")
    print(f"args.channels {args.channels}")
    return(args)


def main(args):

    args = parse_args(args)
    callback_obj = AudioCallback(decibels_target=args.decibels_target,
                                 decibels_upper_limit=args.decibels_upper_limit,
                                 decibels_lower_limit=args.decibels_lower_limit,
                                 interval=args.interval,
                                 samplerate=args.samplerate)
    try:

        with sd.InputStream(device=args.device, channels=max(args.channels),
                            samplerate=args.samplerate, callback=callback_obj):
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
