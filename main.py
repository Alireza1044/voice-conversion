import pyaudio
import numpy as np
import librosa
import scipy
from pyo import *


def pitch_shift(signal, n):
    r = 2.0 ** (-float(n) / 12)
    stft = librosa.core.stft(signal)
    stft_stretch = librosa.core.phase_vocoder(stft, r)
    len_stretch = int(round(len(signal) / r))
    signal_stretch = librosa.core.istft(stft_stretch, dtype=signal.dtype, length=len_stretch)
    n_samples = len(signal)
    signal_resample = scipy.signal.resample(signal_stretch, n_samples, axis=-1)
    signal_resample = librosa.util.fix_length(signal_resample, len(signal))
    return signal_resample


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
    new_signal = pitch_shift(ch, shift_amount)
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
