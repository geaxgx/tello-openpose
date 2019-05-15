"""
    Classes to play sounds and tones on pygame

    class SoundPlayer : manage a FIFO queue to play sounds from ogg files in a dedicated channel.
                        - load(name, filename): method that loads an ogg file 'filename' and associates the name 'name' to that sound
                        - play(name=None): if name not None, enqueue the corresponding sound in the FIFO. If the channel is not busy, play the next sound from the FIFO

    class Tone : to play a sinusoidal wave in a dedicated channel. Methods 'on' and 'off' to play or stop the tone.

"""
import pygame
from time import sleep
import logging
import sys
import numpy as np

log = logging.getLogger("PygameAudio")

class PygameAudio :
    _init = False
    _channels_used = 0
    def __init__(self, sampleRate=22050, debug=False):
        if debug:
            log.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S"))
            log.addHandler(ch)
        print("_channels_used",self._channels_used)
        self.sampleRate = sampleRate
        if not PygameAudio._init:
            log.info(f"mixer init - sampleRate: {sampleRate}")
            pygame.mixer.init(sampleRate, -16, 1, 128)
            PygameAudio._init = True
        self._channel_id = PygameAudio._channels_used
        log.info(f"init channel {self._channel_id}")
        self._channel = pygame.mixer.Channel(self._channel_id)
        PygameAudio._channels_used += 1


class SoundPlayer(PygameAudio):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self._raw_sounds={}
        self._fifo_sounds = []
        self._debug = debug
        
    def load(self, name, filename):
        log.info(f"loading {name} {filename}")
        self._raw_sounds[name] = pygame.mixer.Sound(filename)
    def play(self, name=None):
        if name is not None:
            self._fifo_sounds.append((name,self._raw_sounds[name]))
            log.info(f"queuing '{name}' (remaining: {len(self._fifo_sounds)}) ")
        if len(self._fifo_sounds) > 0:
            if not self._channel.get_busy():
                
                name,sound = self._fifo_sounds.pop(0)
                log.info(f"playing '{name}' on channel {self._channel_id} (remaining: {len(self._fifo_sounds)})")
                self._channel.queue(sound)
        
class Tone(PygameAudio):
    def __init__(self, freq=440, debug=False):
        super().__init__(debug=debug)
        self.freq = freq
        arr = np.array([4096 * np.sin(2.0 * np.pi * 440 * x / self.sampleRate) for x in range(0, self.sampleRate)]).astype(np.int16)
        self.sound = pygame.sndarray.make_sound(arr)

    def on(self):
        log.info(f"play tone {self.freq}Hz on channel {self._channel_id}")
        self._channel.play(self.sound,-1)
    def off(self):
        log.info(f"stop tone {self.freq}Hz on channel {self._channel_id}")
        self._channel.stop()

if __name__ == '__main__':
    import random

    sp = SoundPlayer(debug=True)
    t = Tone()
    sp.load("hello", "sounds/hello.ogg")
    sp.load("bonjour","sounds/bonjour.ogg")
    sp.play("hello")
    sp.play("hello")
    sp.play("hello")
    sp.play("hello")
    
    prev_on = 0
    for i in range(10):
        print(i)
        on = random.randint(0,1)
        if on != prev_on:
            if on: 
                t.on() 
            else: t.off()
            prev_on = on

        if i==2:
            sp.play("bonjour")
            sp.play("bonjour")
        sp.play()
        sleep(2)
    
    if prev_on: t.off()