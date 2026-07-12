"""
Audio DSP chain for VRK Kiosk — EP-06 (Ambient Noise Handling)

Order of operations on every utterance:
  1. bandpass filter   — remove rumble (<80Hz) and hiss (>7.5kHz)
  2. energy gate       — reject too-quiet audio (person not close enough)
  3. spectral denoise  — optional, subtracts profiled ambient noise

All functions operate on 1-D float32 numpy arrays at 16 kHz.
"""

import numpy as np
from scipy.signal import butter, filtfilt, stft, istft

SAMPLE_RATE = 16000

# ---------------------------------------------------------------- bandpass

_BP_CACHE = {}

def _bandpass_coeffs(lowcut=80.0, highcut=7500.0, fs=SAMPLE_RATE, order=5):
    key = (lowcut, highcut, fs, order)
    if key not in _BP_CACHE:                      # compute once, reuse
        nyq = 0.5 * fs
        _BP_CACHE[key] = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return _BP_CACHE[key]


def apply_bandpass(audio: np.ndarray) -> np.ndarray:
    """Keep human-speech band, drop AC hum / fan rumble / electrical hiss."""
    b, a = _bandpass_coeffs()
    return filtfilt(b, a, audio).astype(np.float32)


# ---------------------------------------------------------------- energy gate

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0


def is_too_quiet(audio: np.ndarray, threshold: float = 0.010) -> bool:
    """
    Person standing at the kiosk is 10–15 dB louder than crowd noise.
    Tune `threshold` on-site: log rms() values for real users vs ambient,
    pick a value between the two clusters. 0.010 is a starting point for
    a mic ~40 cm from the speaker.
    """
    return rms(audio) < threshold


# ---------------------------------------------------------------- spectral denoise

class SpectralNoiseReducer:
    """
    Spectral subtraction using a profile of the room's ambient noise.
    Profile it from audio captured while nobody is speaking (e.g. the
    idle-mode mic buffer, or VAD-rejected chunks).
    """

    def __init__(self, n_fft: int = 2048, hop: int = 512):
        self.n_fft = n_fft
        self.hop = hop
        self.noise_profile = None

    def profile_noise(self, noise_audio: np.ndarray) -> None:
        if len(noise_audio) < self.n_fft:
            noise_audio = np.pad(noise_audio, (0, self.n_fft - len(noise_audio)))
        _, _, Z = stft(noise_audio, nperseg=self.n_fft,
                       noverlap=self.n_fft - self.hop)
        self.noise_profile = np.median(np.abs(Z), axis=1)

    def reduce(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_profile is None:
            return audio
        _, _, Z = stft(audio, nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
        mag, phase = np.abs(Z), np.angle(Z)
        clean = np.maximum(mag - self.noise_profile[:, None], mag * 0.1)
        _, out = istft(clean * np.exp(1j * phase),
                       nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
        return out[: len(audio)].astype(np.float32)


# ---------------------------------------------------------------- full chain

def preprocess(audio: np.ndarray,
               denoiser: "SpectralNoiseReducer | None" = None) -> "np.ndarray | None":
    """
    Run the full DSP chain. Returns cleaned audio, or None if the
    utterance should be rejected (too quiet → ask visitor to speak up).
    """
    audio = apply_bandpass(audio)
    if is_too_quiet(audio):
        return None
    if denoiser is not None:
        audio = denoiser.reduce(audio)
    return audio
