#!/usr/bin/env python3
"""
Multi-Track Audio Processing System
Professional audio mixing, effects, and automation
"""

import numpy as np
import soundfile as sf
import librosa
import librosa.display
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import pyloudnorm as pyln
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AudioTrackType(Enum):
    """Types of audio tracks"""
    DIALOGUE = "dialogue"
    MUSIC = "music"
    EFFECTS = "effects"
    AMBIENCE = "ambience"
    VOICEOVER = "voiceover"
    FOLEY = "foley"
    MASTER = "master"

class AudioEffect(Enum):
    """Available audio effects"""
    EQ = "equalizer"
    COMPRESSOR = "compressor"
    LIMITER = "limiter"
    REVERB = "reverb"
    DELAY = "delay"
    CHORUS = "chorus"
    FLANGER = "flanger"
    PHASER = "phaser"
    DISTORTION = "distortion"
    GATE = "gate"
    DEESSER = "de-esser"
    EXCITER = "exciter"
    STEREO_WIDENER = "stereo_widener"
    PITCH_SHIFT = "pitch_shift"
    TIME_STRETCH = "time_stretch"

@dataclass
class AudioTrack:
    """Represents an audio track"""
    name: str
    track_type: AudioTrackType
    audio_data: np.ndarray
    sample_rate: int
    volume: float = 1.0
    pan: float = 0.0  # -1 (left) to 1 (right)
    mute: bool = False
    solo: bool = False
    effects_chain: List[Dict] = None
    automation: Dict[str, List[Tuple[float, float]]] = None  # parameter -> [(time, value)]

@dataclass
class MixBus:
    """Audio mix bus for grouping tracks"""
    name: str
    tracks: List[AudioTrack]
    volume: float = 1.0
    effects_chain: List[Dict] = None
    send_level: float = 0.0  # Send to reverb bus

class NoiseReductionNN(nn.Module):
    """Neural network for noise reduction"""
    
    def __init__(self, n_fft=2048):
        super().__init__()
        self.n_fft = n_fft
        self.input_size = n_fft // 2 + 1
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.input_size),
            nn.Sigmoid()  # Output mask
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        mask = self.decoder(encoded)
        return mask

class MultiTrackAudioProcessor:
    """Professional multi-track audio processing system"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.tracks: List[AudioTrack] = []
        self.buses: List[MixBus] = []
        self.master_bus: Optional[MixBus] = None
        
        # Initialize AI models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.noise_reducer = NoiseReductionNN().to(self.device)
        self.noise_reducer.eval()
        
        # Loudness meter
        self.meter = pyln.Meter(sample_rate)
        
        # Default settings
        self.target_loudness = -23.0  # LUFS for broadcast
        self.headroom = -1.0  # dB
        
    def add_track(
        self,
        audio_file: str,
        name: str,
        track_type: AudioTrackType = AudioTrackType.DIALOGUE
    ) -> AudioTrack:
        """Add audio track from file"""
        
        # Load audio
        audio_data, sr = librosa.load(audio_file, sr=self.sample_rate, mono=False)
        
        # Convert to stereo if mono
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])
        
        track = AudioTrack(
            name=name,
            track_type=track_type,
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            effects_chain=[],
            automation={}
        )
        
        self.tracks.append(track)
        logger.info(f"Added track: {name} ({track_type.value})")
        
        return track
    
    def create_bus(self, name: str, track_indices: List[int]) -> MixBus:
        """Create mix bus from tracks"""
        
        selected_tracks = [self.tracks[i] for i in track_indices if i < len(self.tracks)]
        
        bus = MixBus(
            name=name,
            tracks=selected_tracks,
            effects_chain=[]
        )
        
        self.buses.append(bus)
        logger.info(f"Created bus: {name} with {len(selected_tracks)} tracks")
        
        return bus
    
    def apply_eq(
        self,
        audio: np.ndarray,
        freq_bands: List[float] = [60, 200, 800, 2000, 8000],
        gains: List[float] = [0, 0, 0, 0, 0],
        q_factors: List[float] = [0.7, 0.7, 0.7, 0.7, 0.7]
    ) -> np.ndarray:
        """Apply parametric equalizer"""
        
        result = audio.copy()
        
        for freq, gain, q in zip(freq_bands, gains, q_factors):
            if gain == 0:
                continue
            
            # Design peaking EQ filter
            w0 = 2 * np.pi * freq / self.sample_rate
            A = 10 ** (gain / 40)
            alpha = np.sin(w0) / (2 * q)
            
            # Filter coefficients
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
            # Normalize
            b = [b0/a0, b1/a0, b2/a0]
            a = [1, a1/a0, a2/a0]
            
            # Apply filter
            if audio.ndim == 1:
                result = signal.lfilter(b, a, result)
            else:
                for ch in range(audio.shape[0]):
                    result[ch] = signal.lfilter(b, a, result[ch])
        
        return result
    
    def apply_compressor(
        self,
        audio: np.ndarray,
        threshold: float = -20,  # dB
        ratio: float = 4,
        attack: float = 5,  # ms
        release: float = 100,  # ms
        knee: float = 2,  # dB
        makeup_gain: float = 0  # dB
    ) -> np.ndarray:
        """Apply dynamic range compression"""
        
        # Convert parameters
        threshold_linear = 10 ** (threshold / 20)
        attack_samples = int(attack * self.sample_rate / 1000)
        release_samples = int(release * self.sample_rate / 1000)
        
        # Envelope follower
        if audio.ndim == 1:
            envelope = self._envelope_follower(np.abs(audio), attack_samples, release_samples)
            gain_reduction = self._compute_gain_reduction(
                envelope, threshold_linear, ratio, knee
            )
            result = audio * gain_reduction
        else:
            result = audio.copy()
            for ch in range(audio.shape[0]):
                envelope = self._envelope_follower(np.abs(audio[ch]), attack_samples, release_samples)
                gain_reduction = self._compute_gain_reduction(
                    envelope, threshold_linear, ratio, knee
                )
                result[ch] = audio[ch] * gain_reduction
        
        # Apply makeup gain
        makeup_linear = 10 ** (makeup_gain / 20)
        result *= makeup_linear
        
        return result
    
    def apply_limiter(
        self,
        audio: np.ndarray,
        ceiling: float = -0.3,  # dB
        release: float = 50,  # ms
        lookahead: float = 5  # ms
    ) -> np.ndarray:
        """Apply brick-wall limiter"""
        
        ceiling_linear = 10 ** (ceiling / 20)
        lookahead_samples = int(lookahead * self.sample_rate / 1000)
        release_samples = int(release * self.sample_rate / 1000)
        
        # Look-ahead buffer
        if audio.ndim == 1:
            padded = np.pad(audio, (lookahead_samples, 0), mode='constant')
            envelope = self._peak_envelope(padded, lookahead_samples)
            gain = np.minimum(1.0, ceiling_linear / (envelope + 1e-10))
            gain = self._smooth_gain(gain, release_samples)
            result = audio * gain[:len(audio)]
        else:
            result = audio.copy()
            for ch in range(audio.shape[0]):
                padded = np.pad(audio[ch], (lookahead_samples, 0), mode='constant')
                envelope = self._peak_envelope(padded, lookahead_samples)
                gain = np.minimum(1.0, ceiling_linear / (envelope + 1e-10))
                gain = self._smooth_gain(gain, release_samples)
                result[ch] = audio[ch] * gain[:len(audio[ch])]
        
        return np.clip(result, -ceiling_linear, ceiling_linear)
    
    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7,
        pre_delay: float = 20  # ms
    ) -> np.ndarray:
        """Apply reverb effect"""
        
        # Simple reverb using comb and allpass filters
        pre_delay_samples = int(pre_delay * self.sample_rate / 1000)
        
        # Comb filter delays (in samples)
        comb_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        comb_delays = [int(d * room_size) for d in comb_delays]
        
        # Allpass filter delays
        allpass_delays = [225, 556, 441, 341]
        
        if audio.ndim == 1:
            # Pre-delay
            delayed = np.pad(audio, (pre_delay_samples, 0), mode='constant')[:len(audio)]
            
            # Comb filters
            comb_out = np.zeros_like(audio)
            for delay in comb_delays:
                filtered = self._comb_filter(delayed, delay, damping)
                comb_out += filtered / len(comb_delays)
            
            # Allpass filters
            reverb = comb_out
            for delay in allpass_delays:
                reverb = self._allpass_filter(reverb, delay)
            
            # Mix wet and dry
            result = dry_level * audio + wet_level * reverb
        else:
            result = audio.copy()
            for ch in range(audio.shape[0]):
                delayed = np.pad(audio[ch], (pre_delay_samples, 0), mode='constant')[:len(audio[ch])]
                
                comb_out = np.zeros_like(audio[ch])
                for delay in comb_delays:
                    filtered = self._comb_filter(delayed, delay, damping)
                    comb_out += filtered / len(comb_delays)
                
                reverb = comb_out
                for delay in allpass_delays:
                    reverb = self._allpass_filter(reverb, delay)
                
                result[ch] = dry_level * audio[ch] + wet_level * reverb
        
        return result
    
    def apply_delay(
        self,
        audio: np.ndarray,
        delay_time: float = 250,  # ms
        feedback: float = 0.5,
        mix: float = 0.3
    ) -> np.ndarray:
        """Apply delay effect"""
        
        delay_samples = int(delay_time * self.sample_rate / 1000)
        
        if audio.ndim == 1:
            result = audio.copy()
            delay_buffer = np.zeros(delay_samples)
            
            for i in range(len(audio)):
                delayed = delay_buffer[0]
                delay_buffer = np.roll(delay_buffer, -1)
                delay_buffer[-1] = audio[i] + delayed * feedback
                result[i] = audio[i] * (1 - mix) + delayed * mix
        else:
            result = audio.copy()
            for ch in range(audio.shape[0]):
                delay_buffer = np.zeros(delay_samples)
                
                for i in range(len(audio[ch])):
                    delayed = delay_buffer[0]
                    delay_buffer = np.roll(delay_buffer, -1)
                    delay_buffer[-1] = audio[ch][i] + delayed * feedback
                    result[ch][i] = audio[ch][i] * (1 - mix) + delayed * mix
        
        return result
    
    def apply_gate(
        self,
        audio: np.ndarray,
        threshold: float = -40,  # dB
        attack: float = 1,  # ms
        hold: float = 10,  # ms
        release: float = 100  # ms
    ) -> np.ndarray:
        """Apply noise gate"""
        
        threshold_linear = 10 ** (threshold / 20)
        attack_samples = int(attack * self.sample_rate / 1000)
        hold_samples = int(hold * self.sample_rate / 1000)
        release_samples = int(release * self.sample_rate / 1000)
        
        if audio.ndim == 1:
            envelope = self._envelope_follower(np.abs(audio), 1, release_samples)
            gate = (envelope > threshold_linear).astype(float)
            
            # Smooth gate transitions
            gate = self._smooth_gate(gate, attack_samples, hold_samples, release_samples)
            result = audio * gate
        else:
            result = audio.copy()
            for ch in range(audio.shape[0]):
                envelope = self._envelope_follower(np.abs(audio[ch]), 1, release_samples)
                gate = (envelope > threshold_linear).astype(float)
                gate = self._smooth_gate(gate, attack_samples, hold_samples, release_samples)
                result[ch] = audio[ch] * gate
        
        return result
    
    def apply_deesser(
        self,
        audio: np.ndarray,
        frequency: float = 7000,  # Hz
        threshold: float = -20,  # dB
        ratio: float = 4
    ) -> np.ndarray:
        """Apply de-esser to reduce sibilance"""
        
        # Bandpass filter around sibilant frequency
        nyquist = self.sample_rate / 2
        low = (frequency - 1000) / nyquist
        high = (frequency + 2000) / nyquist
        
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        
        if audio.ndim == 1:
            # Extract sibilant band
            sibilant = signal.sosfilt(sos, audio)
            
            # Compress sibilant band
            compressed_sibilant = self.apply_compressor(
                sibilant, threshold, ratio, 0.1, 10
            )
            
            # Replace sibilant band
            result = audio - sibilant + compressed_sibilant
        else:
            result = audio.copy()
            for ch in range(audio.shape[0]):
                sibilant = signal.sosfilt(sos, audio[ch])
                compressed_sibilant = self.apply_compressor(
                    sibilant, threshold, ratio, 0.1, 10
                )
                result[ch] = audio[ch] - sibilant + compressed_sibilant
        
        return result
    
    def apply_stereo_widener(
        self,
        audio: np.ndarray,
        width: float = 1.5  # 0 = mono, 1 = normal, >1 = wider
    ) -> np.ndarray:
        """Apply stereo widening effect"""
        
        if audio.ndim != 2 or audio.shape[0] != 2:
            logger.warning("Stereo widener requires stereo input")
            return audio
        
        # Convert to M/S (mid/side)
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        
        # Adjust width
        side *= width
        
        # Convert back to L/R
        result = np.zeros_like(audio)
        result[0] = mid + side
        result[1] = mid - side
        
        return result
    
    def apply_pitch_shift(
        self,
        audio: np.ndarray,
        semitones: float
    ) -> np.ndarray:
        """Apply pitch shifting"""
        
        if audio.ndim == 1:
            # Use librosa for pitch shifting
            shifted = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=semitones
            )
            return shifted
        else:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                result[ch] = librosa.effects.pitch_shift(
                    audio[ch], sr=self.sample_rate, n_steps=semitones
                )
            return result
    
    def apply_time_stretch(
        self,
        audio: np.ndarray,
        rate: float  # <1 = slower, >1 = faster
    ) -> np.ndarray:
        """Apply time stretching without pitch change"""
        
        if audio.ndim == 1:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            return stretched
        else:
            channels = []
            for ch in range(audio.shape[0]):
                stretched = librosa.effects.time_stretch(audio[ch], rate=rate)
                channels.append(stretched)
            return np.array(channels)
    
    def apply_automation(
        self,
        audio: np.ndarray,
        automation_points: List[Tuple[float, float]],
        parameter: str = "volume"
    ) -> np.ndarray:
        """Apply parameter automation"""
        
        # Create automation curve
        times = [p[0] for p in automation_points]
        values = [p[1] for p in automation_points]
        
        # Interpolate automation curve
        sample_times = np.arange(len(audio if audio.ndim == 1 else audio[0])) / self.sample_rate
        automation_curve = np.interp(sample_times, times, values)
        
        if parameter == "volume":
            if audio.ndim == 1:
                result = audio * automation_curve
            else:
                result = audio * automation_curve[np.newaxis, :]
        elif parameter == "pan":
            if audio.ndim == 2:
                result = audio.copy()
                # Convert pan (-1 to 1) to L/R gains
                left_gain = np.sqrt((1 - automation_curve) / 2)
                right_gain = np.sqrt((1 + automation_curve) / 2)
                result[0] *= left_gain
                result[1] *= right_gain
            else:
                result = audio
        else:
            result = audio
        
        return result
    
    def mix_tracks(
        self,
        tracks: List[AudioTrack],
        normalize: bool = True
    ) -> np.ndarray:
        """Mix multiple tracks together"""
        
        # Find maximum length
        max_length = max(t.audio_data.shape[-1] for t in tracks)
        
        # Initialize mix buffer (stereo)
        mix = np.zeros((2, max_length))
        
        for track in tracks:
            if track.mute:
                continue
            
            # Get track audio
            track_audio = track.audio_data
            
            # Apply effects chain
            for effect in track.effects_chain or []:
                track_audio = self.apply_effect(track_audio, effect)
            
            # Apply volume
            track_audio *= track.volume
            
            # Apply pan
            if track_audio.ndim == 1:
                # Mono to stereo with pan
                left_gain = np.sqrt((1 - track.pan) / 2)
                right_gain = np.sqrt((1 + track.pan) / 2)
                stereo = np.array([
                    track_audio * left_gain,
                    track_audio * right_gain
                ])
            else:
                stereo = track_audio
            
            # Apply automation
            if track.automation:
                for param, points in track.automation.items():
                    stereo = self.apply_automation(stereo, points, param)
            
            # Add to mix
            length = stereo.shape[-1]
            mix[:, :length] += stereo
        
        # Normalize to prevent clipping
        if normalize:
            max_val = np.max(np.abs(mix))
            if max_val > 1.0:
                mix /= max_val
        
        return mix
    
    def apply_effect(self, audio: np.ndarray, effect_config: Dict) -> np.ndarray:
        """Apply audio effect based on configuration"""
        
        effect_type = effect_config.get("type")
        params = effect_config.get("params", {})
        
        if effect_type == AudioEffect.EQ.value:
            return self.apply_eq(audio, **params)
        elif effect_type == AudioEffect.COMPRESSOR.value:
            return self.apply_compressor(audio, **params)
        elif effect_type == AudioEffect.LIMITER.value:
            return self.apply_limiter(audio, **params)
        elif effect_type == AudioEffect.REVERB.value:
            return self.apply_reverb(audio, **params)
        elif effect_type == AudioEffect.DELAY.value:
            return self.apply_delay(audio, **params)
        elif effect_type == AudioEffect.GATE.value:
            return self.apply_gate(audio, **params)
        elif effect_type == AudioEffect.DEESSER.value:
            return self.apply_deesser(audio, **params)
        elif effect_type == AudioEffect.STEREO_WIDENER.value:
            return self.apply_stereo_widener(audio, **params)
        elif effect_type == AudioEffect.PITCH_SHIFT.value:
            return self.apply_pitch_shift(audio, **params)
        elif effect_type == AudioEffect.TIME_STRETCH.value:
            return self.apply_time_stretch(audio, **params)
        else:
            return audio
    
    def apply_ai_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply AI-based noise reduction"""
        
        # STFT
        if audio.ndim == 1:
            stft = librosa.stft(audio, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply neural network
            mag_tensor = torch.from_numpy(magnitude.T).float().to(self.device)
            
            with torch.no_grad():
                mask = self.noise_reducer(mag_tensor)
                mask = mask.cpu().numpy().T
            
            # Apply mask
            cleaned_magnitude = magnitude * mask
            
            # Inverse STFT
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            result = librosa.istft(cleaned_stft)
        else:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                stft = librosa.stft(audio[ch], n_fft=2048)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                mag_tensor = torch.from_numpy(magnitude.T).float().to(self.device)
                
                with torch.no_grad():
                    mask = self.noise_reducer(mag_tensor)
                    mask = mask.cpu().numpy().T
                
                cleaned_magnitude = magnitude * mask
                cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
                result[ch] = librosa.istft(cleaned_stft)
        
        return result
    
    def auto_mix(self) -> np.ndarray:
        """Automatically mix all tracks with intelligent processing"""
        
        # Group tracks by type
        dialogue_tracks = [t for t in self.tracks if t.track_type == AudioTrackType.DIALOGUE]
        music_tracks = [t for t in self.tracks if t.track_type == AudioTrackType.MUSIC]
        effects_tracks = [t for t in self.tracks if t.track_type == AudioTrackType.EFFECTS]
        
        # Apply appropriate processing to each group
        for track in dialogue_tracks:
            # Add dialogue processing
            track.effects_chain = [
                {"type": AudioEffect.GATE.value, "params": {"threshold": -45}},
                {"type": AudioEffect.EQ.value, "params": {
                    "freq_bands": [80, 200, 2000, 8000],
                    "gains": [-3, 0, 2, 1],
                    "q_factors": [0.7, 0.7, 0.7, 0.7]
                }},
                {"type": AudioEffect.COMPRESSOR.value, "params": {
                    "threshold": -20, "ratio": 3, "attack": 5, "release": 50
                }},
                {"type": AudioEffect.DEESSER.value, "params": {"frequency": 7000}}
            ]
        
        for track in music_tracks:
            # Add music processing
            track.effects_chain = [
                {"type": AudioEffect.EQ.value, "params": {
                    "freq_bands": [60, 200, 1000, 5000, 12000],
                    "gains": [1, 0, -1, 0, 1],
                    "q_factors": [0.7, 0.7, 0.7, 0.7, 0.7]
                }},
                {"type": AudioEffect.COMPRESSOR.value, "params": {
                    "threshold": -15, "ratio": 2, "attack": 10, "release": 100
                }},
                {"type": AudioEffect.STEREO_WIDENER.value, "params": {"width": 1.2}}
            ]
            # Duck music under dialogue
            track.volume = 0.6
        
        for track in effects_tracks:
            # Add effects processing
            track.effects_chain = [
                {"type": AudioEffect.EQ.value, "params": {
                    "freq_bands": [100, 500, 2000, 8000],
                    "gains": [-2, 0, 1, 2],
                    "q_factors": [0.7, 0.7, 0.7, 0.7]
                }}
            ]
            track.volume = 0.8
        
        # Mix all tracks
        mix = self.mix_tracks(self.tracks)
        
        # Master processing
        mix = self.apply_eq(mix, 
            freq_bands=[60, 200, 1000, 5000, 12000],
            gains=[0, 0, 0, 0, 0],
            q_factors=[0.7, 0.7, 0.7, 0.7, 0.7]
        )
        
        mix = self.apply_compressor(mix,
            threshold=-10, ratio=2, attack=10, release=100, makeup_gain=2
        )
        
        mix = self.apply_limiter(mix, ceiling=-0.3)
        
        # Normalize to target loudness
        mix = self.normalize_loudness(mix, self.target_loudness)
        
        return mix
    
    def normalize_loudness(self, audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
        """Normalize audio to target loudness (LUFS)"""
        
        # Measure loudness
        if audio.ndim == 1:
            loudness = self.meter.integrated_loudness(audio)
        else:
            # Convert to mono for measurement
            mono = np.mean(audio, axis=0)
            loudness = self.meter.integrated_loudness(mono)
        
        # Calculate gain needed
        gain_db = target_lufs - loudness
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 0.99:
            normalized *= 0.99 / max_val
        
        logger.info(f"Normalized from {loudness:.1f} to {target_lufs:.1f} LUFS")
        
        return normalized
    
    def export_mix(
        self,
        output_path: str,
        mix: Optional[np.ndarray] = None,
        format: str = 'WAV',
        bit_depth: int = 24
    ):
        """Export mixed audio to file"""
        
        if mix is None:
            mix = self.auto_mix()
        
        # Ensure stereo
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        
        # Transpose for soundfile (samples, channels)
        mix = mix.T
        
        # Write file
        subtype = 'PCM_24' if bit_depth == 24 else 'PCM_16'
        sf.write(output_path, mix, self.sample_rate, subtype=subtype)
        
        logger.info(f"Exported mix to {output_path}")
    
    def apply_ducking(
        self,
        music_track: AudioTrack,
        dialogue_track: AudioTrack,
        threshold: float = -30,  # dB
        duck_amount: float = -10,  # dB
        attack: float = 10,  # ms
        release: float = 100  # ms
    ) -> AudioTrack:
        """Apply automatic ducking (reduce music when dialogue is present)"""
        
        # Get dialogue envelope
        dialogue_mono = np.mean(dialogue_track.audio_data, axis=0) if dialogue_track.audio_data.ndim > 1 else dialogue_track.audio_data
        
        # Detect dialogue presence
        threshold_linear = 10 ** (threshold / 20)
        attack_samples = int(attack * self.sample_rate / 1000)
        release_samples = int(release * self.sample_rate / 1000)
        
        envelope = self._envelope_follower(np.abs(dialogue_mono), attack_samples, release_samples)
        dialogue_present = envelope > threshold_linear
        
        # Create ducking curve
        duck_gain = 10 ** (duck_amount / 20)
        gain_curve = np.ones_like(dialogue_mono)
        gain_curve[dialogue_present] = duck_gain
        
        # Smooth gain curve
        gain_curve = self._smooth_gain(gain_curve, release_samples)
        
        # Apply to music track
        music_track.automation["volume"] = [
            (i / self.sample_rate, gain_curve[i])
            for i in range(0, len(gain_curve), self.sample_rate // 10)  # 10 points per second
        ]
        
        return music_track
    
    def _envelope_follower(
        self,
        signal: np.ndarray,
        attack_samples: int,
        release_samples: int
    ) -> np.ndarray:
        """Track signal envelope"""
        
        envelope = np.zeros_like(signal)
        
        for i in range(1, len(signal)):
            if signal[i] > envelope[i-1]:
                # Attack
                alpha = 1.0 - np.exp(-1.0 / attack_samples) if attack_samples > 0 else 1.0
                envelope[i] = signal[i] * alpha + envelope[i-1] * (1 - alpha)
            else:
                # Release
                alpha = 1.0 - np.exp(-1.0 / release_samples) if release_samples > 0 else 1.0
                envelope[i] = signal[i] * alpha + envelope[i-1] * (1 - alpha)
        
        return envelope
    
    def _compute_gain_reduction(
        self,
        envelope: np.ndarray,
        threshold: float,
        ratio: float,
        knee: float
    ) -> np.ndarray:
        """Compute compressor gain reduction"""
        
        gain = np.ones_like(envelope)
        
        # Soft knee compression
        for i, env in enumerate(envelope):
            if env > threshold:
                # Above threshold
                excess = env - threshold
                
                if knee > 0 and excess < knee:
                    # Soft knee region
                    knee_ratio = 1 + (ratio - 1) * (excess / knee) ** 2
                    gain[i] = threshold + excess / knee_ratio
                else:
                    # Hard knee
                    gain[i] = threshold + excess / ratio
                
                gain[i] = gain[i] / env if env > 0 else 1.0
        
        return gain
    
    def _peak_envelope(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Get peak envelope with lookahead"""
        
        envelope = np.zeros_like(signal)
        
        for i in range(len(signal)):
            start = max(0, i)
            end = min(len(signal), i + window)
            envelope[i] = np.max(np.abs(signal[start:end]))
        
        return envelope
    
    def _smooth_gain(self, gain: np.ndarray, window: int) -> np.ndarray:
        """Smooth gain changes"""
        
        if window > 1:
            kernel = np.ones(window) / window
            gain = np.convolve(gain, kernel, mode='same')
        
        return gain
    
    def _smooth_gate(
        self,
        gate: np.ndarray,
        attack: int,
        hold: int,
        release: int
    ) -> np.ndarray:
        """Smooth gate transitions"""
        
        result = gate.copy()
        hold_counter = 0
        
        for i in range(1, len(gate)):
            if gate[i] > gate[i-1]:
                # Opening - attack
                alpha = 1.0 - np.exp(-1.0 / attack) if attack > 0 else 1.0
                result[i] = result[i-1] + alpha * (1 - result[i-1])
            elif gate[i] < gate[i-1]:
                # Potentially closing
                if hold_counter < hold:
                    # Hold open
                    result[i] = result[i-1]
                    hold_counter += 1
                else:
                    # Release
                    alpha = 1.0 - np.exp(-1.0 / release) if release > 0 else 1.0
                    result[i] = result[i-1] * (1 - alpha)
            else:
                # Stable
                if gate[i] == 1:
                    hold_counter = 0
        
        return result
    
    def _comb_filter(
        self,
        signal: np.ndarray,
        delay: int,
        damping: float
    ) -> np.ndarray:
        """Comb filter for reverb"""
        
        output = np.zeros_like(signal)
        buffer = np.zeros(delay)
        
        for i in range(len(signal)):
            output[i] = signal[i] + damping * buffer[0]
            buffer = np.roll(buffer, -1)
            buffer[-1] = output[i]
        
        return output
    
    def _allpass_filter(
        self,
        signal: np.ndarray,
        delay: int,
        gain: float = 0.5
    ) -> np.ndarray:
        """Allpass filter for reverb"""
        
        output = np.zeros_like(signal)
        buffer = np.zeros(delay)
        
        for i in range(len(signal)):
            delayed = buffer[0]
            output[i] = -signal[i] + delayed
            buffer = np.roll(buffer, -1)
            buffer[-1] = signal[i] + gain * delayed
        
        return output

# Integration function
def process_multitrack_audio(
    audio_files: List[str],
    output_path: str,
    auto_mix: bool = True
) -> str:
    """Process multiple audio tracks"""
    
    processor = MultiTrackAudioProcessor()
    
    # Add tracks
    for i, file in enumerate(audio_files):
        track_type = AudioTrackType.DIALOGUE if i == 0 else AudioTrackType.MUSIC
        processor.add_track(file, f"Track {i+1}", track_type)
    
    # Mix
    if auto_mix:
        mix = processor.auto_mix()
    else:
        mix = processor.mix_tracks(processor.tracks)
    
    # Export
    processor.export_mix(output_path, mix)
    
    return output_path