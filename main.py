import os
import wave
import time
import pyaudio
import pitch as pt
from scipy.io import wavfile
import matplotlib.pyplot as plt
import simpleaudio as sa
from playsound import playsound
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import pyo
import threading
from pyo import *
import Recorder


def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round(np.arange(0, len(sound_array), factor))
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[indices.astype(int)]


def stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(sound_array) / f + window_size) + 1)

    for i in np.arange(0, len(sound_array) - (window_size + h), int(h * f)):
        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2 / s1)) % 2 * np.pi
        a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))

        # add to result
        i2 = int(i / f)
        result[i2: i2 + window_size] += hanning_window * np.absolute(a2_rephased)

    result = ((2 ** (16 - 4)) * result / result.max())  # normalize (16bit)

    return result.astype('float32')


def pitch_shift(signal, rate, n):
    return librosa.effects.pitch_shift(signal, rate, n, res_type='kaiser_fast')


def pitchshift(snd_array, n, window_size=2 ** 13, h=2 ** 11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2 ** (1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0 / factor, window_size, h)
    return speedx(stretched[window_size:], factor)


SHORT_NORMALIZE = (1.0 / 32768.0)
chunk = 128
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
swidth = 2
f_name_directory = r'records'
TIMEOUT_LENGTH = 0.1


def callback(in_data, frame_count, time_info, flag):
    ch = np.fromstring(in_data, dtype=np.float32)
    new_signal = pitch_shift(ch, RATE, shift_amount)
    return new_signal, pyaudio.paContinue


if __name__ == '__main__':
    print("Preparing...")

    RATE = 44100
    CHUNK = int(4096 * 3)
    shift_amount = 5

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    stream.start_stream()
    t = time.time()
    while time.time() - t < 3 * stream.get_input_latency():
        pass
    print("Voice Converter Activated...")
    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("Keyboard interrupt detected.")
        print("exiting...")

    stream.stop_stream()
    stream.close()
    p.terminate()
    pass
