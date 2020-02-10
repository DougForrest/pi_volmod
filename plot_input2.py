#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import collections
import sys
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import queue


import utils


data_folder = os.path.join('data')


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def to_decibles(x, scale=1):
    return(20 * np.log10(np.sqrt(np.mean((x * scale) ** 2))))



class AudioCallback:
    def __init__(self, interval=4800, target_decibles=60, decibles_upper_limit=67,
                 decibles_lower_limit=53, scale=32767):
        self.tot_sum = 0
        self.tot_len = 0
        self.mean = 0
        self.interval = interval
        self.calls = 0
        self.scale = scale
        self.current_indata = None
        self.rolling_data = np.array([])
        self.target_decibles = target_decibles
        self.decibles_lst = collections.deque(maxlen=50)
        self.decibles_upper_limit = decibles_upper_limit
        self.decibles_lower_limit = decibles_lower_limit
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
            self.decibles_lst.append(to_decibles(self.rolling_data, scale=self.scale))

            print(f"current decibles {self.decibles_lst[-1]}")

        volume_adjustment = 0
        if self.decibles_lst and self.decibles_lst[-1] > self.decibles_upper_limit:
            volume_adjustment = 'down'

            print('KEY_VOLUMEDOWN')

        if self.decibles_lst and self.decibles_lst[-1] < self.decibles_lower_limit:
            volume_adjustment = 'up'

            print('KEY_VOLUMEUP')

        self.volume_adjustment_lst.append(self.volume_mapping.get(volume_adjustment, 0))
        if volume_adjustment != 0:
            self.adjust_volume(volume_adjustment)
        # import pdb
        # pdb.set_trace()

        if self.calls > 500:
            raise Exception("END")

    def adjust_volume(self, direction):

        if (sum(self.volume_adjustment_lst) < 5) and (sum(self.volume_adjustment_lst) > -5):
            print(f'adjusting volume {direction}')

            self.current_volume += self.volume_mapping[direction]
            print(f"volume estimate {self.current_volume}")



def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    # if status:
        # print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    print(indata)


    # a = indata[::args.downsample, mapping]
    print(indata[::args.downsample, mapping])

    df = pd.DataFrame(indata, columns=['indata'])
    # print(len(a))
    import pdb
    pdb.set_trace()
    # raise Exception("END")
    q.put(indata[::args.downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


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

      # Channel numbers start with 1

    q = queue.Queue()
    try:
        length = int(args.window *  args.samplerate / (1000 * args.downsample))
        print(f'length {length}')

        plotdata = np.zeros((length, len(args.channels)))

        fig, ax = plt.subplots()
        lines = ax.plot(plotdata)
        if len(args.channels) > 1:
            ax.legend(['channel {}'.format(c) for c in args.channels],
                      loc='lower left', ncol=len(args.channels))
        ax.axis((0, len(plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)

        stream = sd.InputStream(
            device=args.device, channels=max(args.channels),
            samplerate=args.samplerate, callback=callback_obj)
        ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
        with stream:
            plt.show()
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))


if __name__ == '__main__':
    main(sys.argv[1:])
