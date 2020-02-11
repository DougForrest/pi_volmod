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
                 decibels_upper_limit=67,
                 decibels_lower_limit=40,
                 scale=32767):

        self.samplerate = samplerate
        self.interval = interval
        self.scale = scale
        self.current_indata = None
        self.rolling_data = np.array([])
        self.current_decible = None
        self.decibels_upper_limit = decibels_upper_limit
        self.decibels_lower_limit = decibels_lower_limit
        self.current_volume = 5
        self.volume_mapping = {'up': 1, 'down': -1}
        self.volume_adjustment_max_retries = 3
        self.volume_adjustment_lst = collections.deque(maxlen=self.volume_adjustment_max_retries)
        self.volume_retries = 0
        self.last_retry = None
        self.total_frames = 0
        self.volume_milliseconds_delay = 100
        self.frames_interval = int((self.samplerate / 1000) * self.interval)

    def __call__(self, indata, frames, time, status):
        self.total_frames += frames
        self.current_indata = indata.copy().flatten()
        self.rolling_data = np.concatenate((self.rolling_data,
                                            self.current_indata), axis=0)

        if self.total_frames >= self.frames_interval:
            self.rolling_data = self.rolling_data[-self.frames_interval:]
            self.total_frames = 0
            self.evalute_volume()


    def evalute_volume(self):
        self.current_decible = to_decibels(self.rolling_data, scale=self.scale)

        print(f"current decibels {self.current_decible}")

        volume_adjustment = 0
        if self.current_decible and self.current_decible > self.decibels_upper_limit:
            volume_adjustment = 'down'

        if self.current_decible and self.current_decible < self.decibels_lower_limit:
            volume_adjustment = 'up'

        self.volume_adjustment_lst.append(self.volume_mapping.get(volume_adjustment, 0))

        if volume_adjustment != 0:
            self.adjust_volume(volume_adjustment)

    def is_volume_adjustment_limit(self):
        if abs(sum(self.volume_adjustment_lst)) == self.volume_adjustment_max_retries:
            return (True)

        return (False)

    def calc_ms_decay(self, x):
        return (2 ** (x + 3) * 100)

    def retry_with_decay(self):
        now = datetime.now()

        if not self.last_retry:
            self.last_retry = now
            self.volume_retries += 1

        milliseconds_past = (now - self.last_retry).total_seconds() * 1000
        print(f'milliseconds_past {milliseconds_past}')

        if milliseconds_past >= self.volume_milliseconds_delay:
            self.volume_retries += 1
            self.last_retry = now
            self.volume_milliseconds_delay = self.calc_ms_decay(self.volume_retries)
            print(f"self.volume_retries {self.volume_retries}")
            print(f"self.volume_milliseconds_delay {self.volume_milliseconds_delay}")
            print(f"self.volume_adjustment_lst {self.volume_adjustment_lst}")
            self.volume_adjustment_lst.pop()
            self.volume_adjustment_lst.pop()
            print(f"self.volume_adjustment_lst {self.volume_adjustment_lst}")


    def adjust_volume(self, direction):

        print(f"adjust_volume self.volume_adjustment_lst {self.volume_adjustment_lst}")
        if self.is_volume_adjustment_limit():
            print('at volume adjustment limit')
            self.retry_with_decay()
            return (None)

        if abs(sum(self.volume_adjustment_lst)) < self.volume_adjustment_max_retries - 1:
            print('reset retries')
            self.volume_retries = 0

        print(f'adjusting volume {direction}')

        self.current_volume += self.volume_mapping[direction]
        print(f"volume estimate {self.current_volume}")

        if cfg.environment == "prod":
            subprocess.Popen(cfg.key_map[direction].split(' '))


def parse_args(args):

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
        help='input channels to plot (default: the first)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-w', '--window', type=float, default=200, metavar='DURATION',
        help='visible time slot (default: %(default)s ms)')
    parser.add_argument(
        '-i', '--interval', type=float, default=300,
        help='rolling window interval size (in milliseconds')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', type=float, help='sampling rate of audio device')
    parser.add_argument(
        '-ul', '--decibels_upper_limit', type=float, default=67, help='decibels upper limit')
    parser.add_argument(
        '-ll', '--decibels_lower_limit', type=float, default=40, help='decibels lower limit')
    parser.add_argument(
        '-n', '--downsample', type=int, default=10, metavar='N',
        help='display every Nth sample (default: %(default)s)')
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
    callback_obj = AudioCallback(decibels_upper_limit=args.decibels_upper_limit,
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
