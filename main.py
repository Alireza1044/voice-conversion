from scipy.io import wavfile
import simpleaudio as sa
import sounddevice as sd
import soundfile as sf
import numpy as np
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

    return result.astype('int16')


def pitchshift(snd_array, n, window_size=2 ** 13, h=2 ** 11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2 ** (1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0 / factor, window_size, h)
    return speedx(stretched[window_size:], factor)


if __name__ == '__main__':
    a = Recorder.Recorder()
    print('Listening beginning')
    shifted_filename = 'records/result.wav'
    filename = "records/voice.wav"
    while True:
        if a.listen():
            print('Playing recorded voice...')
            rate, signal = wavfile.read(filename)
            sd.play(signal, rate)
            status = sd.wait()
            print('Processing...')
            new_signal = pitchshift(signal, 3)
            wavfile.write(shifted_filename, rate, new_signal)
            print('Done, Now playing...')
            data, fs = sf.read(shifted_filename, dtype='float32')
            sd.play(data, fs)
            status = sd.wait()
            print('Returning to listening')
    pass
