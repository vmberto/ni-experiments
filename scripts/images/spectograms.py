from layers.audio_randaugment import AudioRandAugment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram


def plot_spectrogram(audio, sr, title, subplot_index):
    f, t, Sxx = spectrogram(audio, fs=sr)
    plt.subplot(1, 2, subplot_index)  # <- Horizontal layout
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(title)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")


def main():
    # Replace with the path to your .wav file
    wav_path = "../../dataset/mimii_dataset/fan/id_00/normal/00000000.wav"

    sr, audio = wavfile.read(wav_path)

    # Normalize if integer
    if audio.dtype != np.float32:
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val

    # If stereo, convert to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    augmenter = AudioRandAugment(n_ops=2, magnitude_range=(.3, .3))
    augmented_audio = augmenter(audio, sr)

    plt.figure(figsize=(14, 5))  # Wider to accommodate side-by-side plots
    plot_spectrogram(audio, sr, "Original Audio Spectrogram", 1)
    plot_spectrogram(augmented_audio, sr, "Augmented Audio Spectrogram", 2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
