import os

environment = os.environ.get('VOLMOD')

VOLUME_UP = 'irsend SEND_ONCE edifier KEY_VOLUMEUP'
VOLUME_DOWN = 'irsend SEND_ONCE edifier KEY_VOLUMEDOWN'

key_map = {'up': VOLUME_UP, 'down': VOLUME_DOWN}