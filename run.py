#!/usr/bin/env python3

import argparse
import collections
import sys
import os
import numpy as np
import sounddevice as sd
import subprocess


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
    def __init__(self, interval=4800,
                 target_decibels=60,
                 decibels_upper_limit=67,
                 decibels_lower_limit=40,
                 scale=32767):

        self.tot_sum = 0
        self.tot_len = 0
        self.mean = 0
        self.interval = interval
        self.calls = 0
        self.scale = scale
        self.current_indata = None
        self.rolling_data = np.array([])
        self.target_decibels = target_decibels
        self.decibels_lst = collections.deque(maxlen=50)
        self.decibels_upper_limit = decibels_upper_limit
        self.decibels_lower_limit = decibels_lower_limit
        self.current_volume = 5
        self.volume_mapping = {'up': 1, 'down': -1}
        self.volume_adjustment_lst = collections.deque(maxlen=5)

    def __call__(self, indata, frames, time, status):

        self.calls += 1

        self.current_indata = indata.copy().flatten()

        if len(self.current_indata) < self.interval:
            self.rolling_data = np.concatenate((self.rolling_data,
                                                self.current_indata), axis=0)

        if len(self.rolling_data) >= self.interval:

            self.rolling_data = self.rolling_data[-self.interval:]
            self.decibels_lst.append(to_decibels(self.rolling_data,
                                                 scale=self.scale))

            print(f"current decibels {self.decibels_lst[-1]}")

        volume_adjustment = 0
        if self.decibels_lst and self.decibels_lst[-1] > self.decibels_upper_limit:
            volume_adjustment = 'down'

        if self.decibels_lst and self.decibels_lst[-1] < self.decibels_lower_limit:
            volume_adjustment = 'up'

        self.volume_adjustment_lst.append(self.volume_mapping.get(volume_adjustment, 0))
        if volume_adjustment != 0:
            self.adjust_volume(volume_adjustment)

    def adjust_volume(self, direction):

        if (sum(self.volume_adjustment_lst) < 5) and (sum(self.volume_adjustment_lst) > -5):
            print(f'adjusting volume {direction}')

            self.current_volume += self.volume_mapping[direction]
            print(f"volume estimate {self.current_volume}")

            if cfg.environment == "prod":
                subprocess.Popen(cfg.key_map[direction].split(' '), stdout=subprocess.PIPE).communicate()


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
        '-i', '--interval', type=float, default=30,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', type=float, help='sampling rate of audio device')
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
    callback_obj = AudioCallback()
    args = parse_args(args)

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
