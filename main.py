import os
import wave
import time
import pyaudio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import simpleaudio as sa
from playsound import playsound
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
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


def pitch(signal, rate, n):
    # w = 2 * np.pi * 700 / rate
    # out = signal * np.cos(w * np.arange(0, len(signal)))
    # return out
    return librosa.effects.pitch_shift(signal, rate, n, res_type='kaiser_best')


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

if __name__ == '__main__':
    import pyaudio
    import numpy as np

    RATE = 44100
    CHUNK = 128

    p = pyaudio.PyAudio()
    print("Preparing...")
    player = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True,
                    frames_per_buffer=CHUNK)
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    rate, signal = wavfile.read("records/voice.wav")
    print("Recorder activated...")
    while True:
        try:
            signal = stream.read(CHUNK, exception_on_overflow=False)
            ch = np.fromstring(signal, dtype=np.float32)
            new_signal = pitch(ch, RATE, -3)
            player.write(new_signal, CHUNK,
                         exception_on_underflow=False)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected.")
            print("exiting...")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    pass
