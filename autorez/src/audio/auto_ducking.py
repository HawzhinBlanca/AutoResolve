#!/usr/bin/env python3
"""
Automatic Audio Ducking System
Intelligent ducking with speech detection and music analysis
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DuckingMode(Enum):
    """Ducking operation modes"""
    DIALOGUE = "dialogue"          # Duck music under dialogue
    VOICEOVER = "voiceover"        # Duck all under voiceover
    MUSIC = "music"                # Duck other music
    EFFECTS = "effects"            # Duck under sound effects
    SIDECHAIN = "sidechain"        # Traditional sidechain compression
    ADAPTIVE = "adaptive"          # AI-based adaptive ducking

class DetectionMethod(Enum):
    """Audio detection methods"""
    ENERGY = "energy"              # Energy-based detection
    SPECTRAL = "spectral"          # Spectral characteristics
    VOICE_ACTIVITY = "vad"        # Voice activity detection
    ML_BASED = "ml"                # Machine learning detection
    FREQUENCY = "frequency"        # Frequency-based detection

@dataclass
class DuckingProfile:
    """Ducking behavior profile"""
    threshold_db: float = -20.0       # Trigger threshold
    ratio: float = 4.0                 # Ducking ratio
    attack_ms: float = 10.0           # Attack time
    hold_ms: float = 50.0             # Hold time
    release_ms: float = 100.0         # Release time
    duck_amount_db: float = -12.0     # Maximum duck amount
    lookahead_ms: float = 5.0         # Lookahead time
    frequency_range: Optional[Tuple[float, float]] = None  # Freq range to detect
    sidechain_filter: bool = False    # Apply filter to sidechain

@dataclass
class DuckingRegion:
    """Region where ducking is applied"""
    start_sample: int
    end_sample: int
    duck_amount: float
    fade_in_samples: int
    fade_out_samples: int
    reason: str

class AudioAnalyzer:
    """Analyze audio for ducking triggers"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
    def detect_speech(self, audio: np.ndarray, method: DetectionMethod = DetectionMethod.VOICE_ACTIVITY) -> np.ndarray:
        """Detect speech regions in audio"""
        if method == DetectionMethod.ENERGY:
            return self._detect_energy(audio, threshold_percentile=75)
        elif method == DetectionMethod.SPECTRAL:
            return self._detect_spectral_speech(audio)
        elif method == DetectionMethod.VOICE_ACTIVITY:
            return self._detect_vad(audio)
        else:
            return self._detect_energy(audio)
    
    def _detect_energy(self, audio: np.ndarray, threshold_percentile: float = 75) -> np.ndarray:
        """Energy-based detection"""
        # Calculate RMS energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)     # 10ms hop
        
        # Pad audio
        audio_padded = np.pad(audio, (frame_length // 2, frame_length // 2), mode='constant')
        
        # Calculate RMS for each frame
        rms = []
        for i in range(0, len(audio_padded) - frame_length, hop_length):
            frame = audio_padded[i:i + frame_length]
            rms.append(np.sqrt(np.mean(frame ** 2)))
        
        rms = np.array(rms)
        
        # Dynamic threshold
        threshold = np.percentile(rms, threshold_percentile)
        
        # Create detection array
        detection = rms > threshold
        
        # Expand to sample resolution
        detection_samples = np.zeros(len(audio), dtype=bool)
        for i, det in enumerate(detection):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            detection_samples[start:end] = det
        
        return detection_samples
    
    def _detect_spectral_speech(self, audio: np.ndarray) -> np.ndarray:
        """Detect speech using spectral characteristics"""
        # Spectral centroid and rolloff
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # Speech typically has centroid between 200-4000 Hz
        speech_mask = (spectral_centroid > 200) & (spectral_centroid < 4000)
        
        # Expand to sample resolution
        hop_length = 512
        detection_samples = np.zeros(len(audio), dtype=bool)
        
        for i, is_speech in enumerate(speech_mask):
            start = i * hop_length
            end = min(start + hop_length, len(audio))
            detection_samples[start:end] = is_speech
        
        return detection_samples
    
    def _detect_vad(self, audio: np.ndarray) -> np.ndarray:
        """Voice Activity Detection"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Energy
        energy = librosa.feature.rms(y=audio)[0]
        
        # Combine features
        zcr_norm = (zcr - np.mean(zcr)) / np.std(zcr)
        energy_norm = (energy - np.mean(energy)) / np.std(energy)
        
        # Simple VAD: high energy and moderate ZCR
        vad_score = energy_norm - 0.5 * np.abs(zcr_norm)
        detection = vad_score > 0
        
        # Expand to samples
        hop_length = 512
        detection_samples = np.zeros(len(audio), dtype=bool)
        
        for i, is_voice in enumerate(detection):
            start = i * hop_length
            end = min(start + hop_length, len(audio))
            detection_samples[start:end] = is_voice
        
        # Apply median filter to smooth
        from scipy.ndimage import median_filter
        window_samples = int(0.1 * self.sample_rate)  # 100ms window
        detection_samples = median_filter(detection_samples.astype(float), size=window_samples) > 0.5
        
        return detection_samples
    
    def analyze_frequency_content(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze frequency content of audio"""
        # Compute spectrum
        freqs, times, Sxx = signal.spectrogram(audio, fs=self.sample_rate, nperseg=2048)
        
        # Frequency bands
        bass = np.mean(Sxx[(freqs >= 20) & (freqs < 250)])
        mid = np.mean(Sxx[(freqs >= 250) & (freqs < 4000)])
        high = np.mean(Sxx[(freqs >= 4000) & (freqs < 8000)])
        
        return {
            "bass_energy": bass,
            "mid_energy": mid,
            "high_energy": high,
            "spectral_centroid": np.sum(freqs[:, np.newaxis] * Sxx) / np.sum(Sxx)
        }

class DuckingProcessor:
    """Process audio ducking"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.analyzer = AudioAnalyzer(sample_rate)
        
    def calculate_ducking_envelope(
        self,
        trigger_signal: np.ndarray,
        profile: DuckingProfile
    ) -> np.ndarray:
        """Calculate ducking envelope from trigger signal"""
        # Convert times to samples
        attack_samples = int(profile.attack_ms * self.sample_rate / 1000)
        hold_samples = int(profile.hold_ms * self.sample_rate / 1000)
        release_samples = int(profile.release_ms * self.sample_rate / 1000)
        lookahead_samples = int(profile.lookahead_ms * self.sample_rate / 1000)
        
        # Detect trigger points
        trigger_detection = self._detect_triggers(trigger_signal, profile.threshold_db)
        
        # Apply lookahead
        if lookahead_samples > 0:
            trigger_detection = np.roll(trigger_detection, -lookahead_samples)
            trigger_detection[-lookahead_samples:] = False
        
        # Generate envelope
        envelope = np.ones(len(trigger_signal))
        
        i = 0
        while i < len(trigger_detection):
            if trigger_detection[i]:
                # Find region end
                region_end = i
                while region_end < len(trigger_detection) and trigger_detection[region_end]:
                    region_end += 1
                
                # Apply attack
                attack_end = min(i + attack_samples, region_end)
                for j in range(i, attack_end):
                    progress = (j - i) / attack_samples
                    envelope[j] = 1.0 - progress * (1.0 - db_to_linear(profile.duck_amount_db))
                
                # Apply hold
                hold_end = min(region_end + hold_samples, len(envelope))
                for j in range(attack_end, hold_end):
                    envelope[j] = db_to_linear(profile.duck_amount_db)
                
                # Apply release
                release_end = min(hold_end + release_samples, len(envelope))
                for j in range(hold_end, release_end):
                    progress = (j - hold_end) / release_samples
                    envelope[j] = db_to_linear(profile.duck_amount_db) + progress * (1.0 - db_to_linear(profile.duck_amount_db))
                
                i = region_end
            else:
                i += 1
        
        return envelope
    
    def _detect_triggers(self, signal: np.ndarray, threshold_db: float) -> np.ndarray:
        """Detect trigger points in signal"""
        # Calculate RMS in dB
        frame_length = int(0.010 * self.sample_rate)  # 10ms frames
        
        detection = np.zeros(len(signal), dtype=bool)
        
        for i in range(0, len(signal) - frame_length, frame_length):
            frame = signal[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            
            if rms > 0:
                level_db = 20 * np.log10(rms)
                if level_db > threshold_db:
                    detection[i:i + frame_length] = True
        
        return detection
    
    def apply_ducking(
        self,
        target_audio: np.ndarray,
        trigger_audio: np.ndarray,
        profile: DuckingProfile
    ) -> np.ndarray:
        """Apply ducking to target audio based on trigger"""
        # Calculate ducking envelope
        envelope = self.calculate_ducking_envelope(trigger_audio, profile)
        
        # Apply envelope to target
        if len(target_audio) != len(envelope):
            # Resample envelope if needed
            envelope = np.interp(
                np.linspace(0, len(envelope), len(target_audio)),
                np.arange(len(envelope)),
                envelope
            )
        
        return target_audio * envelope
    
    def adaptive_ducking(
        self,
        music: np.ndarray,
        dialogue: np.ndarray,
        target_loudness_lufs: float = -23.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adaptive ducking based on content analysis"""
        # Analyze dialogue
        dialogue_presence = self.analyzer.detect_speech(dialogue)
        
        # Analyze music frequency content
        music_freq = self.analyzer.analyze_frequency_content(music)
        
        # Create adaptive profile
        profile = DuckingProfile()
        
        # Adjust ducking based on frequency overlap
        if music_freq["mid_energy"] > music_freq["bass_energy"]:
            # More aggressive ducking for mid-heavy music
            profile.duck_amount_db = -15.0
        else:
            # Less ducking for bass-heavy music
            profile.duck_amount_db = -10.0
        
        # Apply ducking
        music_ducked = np.copy(music)
        
        # Process in segments
        segment_length = int(0.5 * self.sample_rate)  # 500ms segments
        
        for i in range(0, len(music), segment_length):
            segment_end = min(i + segment_length, len(music))
            
            if np.any(dialogue_presence[i:segment_end]):
                # Calculate optimal duck amount
                dialogue_segment = dialogue[i:segment_end]
                music_segment = music[i:segment_end]
                
                # Measure levels
                dialogue_level = np.sqrt(np.mean(dialogue_segment ** 2))
                music_level = np.sqrt(np.mean(music_segment ** 2))
                
                if music_level > 0:
                    # Calculate required ducking
                    desired_ratio = 0.3  # Music at 30% of dialogue
                    current_ratio = music_level / (dialogue_level + 1e-10)
                    
                    if current_ratio > desired_ratio:
                        duck_factor = desired_ratio / current_ratio
                        music_ducked[i:segment_end] *= duck_factor
        
        # Smooth transitions
        music_ducked = self._smooth_transitions(music_ducked, window_size=int(0.05 * self.sample_rate))
        
        return music_ducked, dialogue

    def _smooth_transitions(self, audio: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth audio transitions"""
        # Apply moving average filter
        kernel = np.ones(window_size) / window_size
        
        # Pad audio
        audio_padded = np.pad(audio, (window_size // 2, window_size // 2), mode='edge')
        
        # Convolve
        smoothed = np.convolve(audio_padded, kernel, mode='valid')
        
        return smoothed

class AutoDuckingSystem:
    """Complete auto-ducking system"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.processor = DuckingProcessor(sample_rate)
        self.profiles = self._load_default_profiles()
        
    def _load_default_profiles(self) -> Dict[str, DuckingProfile]:
        """Load default ducking profiles"""
        return {
            "gentle": DuckingProfile(
                threshold_db=-25,
                duck_amount_db=-6,
                attack_ms=50,
                release_ms=200
            ),
            "moderate": DuckingProfile(
                threshold_db=-20,
                duck_amount_db=-12,
                attack_ms=20,
                release_ms=150
            ),
            "aggressive": DuckingProfile(
                threshold_db=-15,
                duck_amount_db=-18,
                attack_ms=5,
                release_ms=100
            ),
            "sidechain": DuckingProfile(
                threshold_db=-10,
                duck_amount_db=-20,
                attack_ms=1,
                release_ms=50,
                ratio=8.0
            ),
            "voiceover": DuckingProfile(
                threshold_db=-30,
                duck_amount_db=-15,
                attack_ms=10,
                hold_ms=100,
                release_ms=300
            )
        }
    
    def process_multitrack(
        self,
        tracks: Dict[str, np.ndarray],
        ducking_matrix: Dict[Tuple[str, str], str]
    ) -> Dict[str, np.ndarray]:
        """Process multiple tracks with ducking relationships
        
        Args:
            tracks: Dictionary of track_name -> audio
            ducking_matrix: Dictionary of (trigger, target) -> profile_name
        """
        processed_tracks = {}
        
        # Process each ducking relationship
        for (trigger_name, target_name), profile_name in ducking_matrix.items():
            if trigger_name not in tracks or target_name not in tracks:
                continue
            
            trigger = tracks[trigger_name]
            target = processed_tracks.get(target_name, tracks[target_name])
            
            profile = self.profiles.get(profile_name, self.profiles["moderate"])
            
            # Apply ducking
            ducked = self.processor.apply_ducking(target, trigger, profile)
            processed_tracks[target_name] = ducked
        
        # Copy unprocessed tracks
        for track_name, audio in tracks.items():
            if track_name not in processed_tracks:
                processed_tracks[track_name] = audio
        
        return processed_tracks
    
    def auto_duck_dialogue(
        self,
        dialogue: np.ndarray,
        music: np.ndarray,
        sound_effects: Optional[np.ndarray] = None,
        profile: str = "moderate"
    ) -> Dict[str, np.ndarray]:
        """Automatically duck music and effects under dialogue"""
        result = {"dialogue": dialogue}
        
        ducking_profile = self.profiles.get(profile, self.profiles["moderate"])
        
        # Duck music under dialogue
        result["music"] = self.processor.apply_ducking(music, dialogue, ducking_profile)
        
        # Duck sound effects if provided
        if sound_effects is not None:
            # Use gentler ducking for effects
            effects_profile = DuckingProfile(
                threshold_db=ducking_profile.threshold_db - 5,
                duck_amount_db=ducking_profile.duck_amount_db / 2,
                attack_ms=ducking_profile.attack_ms,
                release_ms=ducking_profile.release_ms
            )
            result["effects"] = self.processor.apply_ducking(sound_effects, dialogue, effects_profile)
        
        return result
    
    def create_sidechain_compression(
        self,
        signal: np.ndarray,
        trigger: np.ndarray,
        frequency: Optional[float] = None
    ) -> np.ndarray:
        """Create sidechain compression effect"""
        if frequency:
            # Create rhythmic trigger
            trigger = self._create_rhythmic_trigger(len(signal), frequency)
        
        profile = self.profiles["sidechain"]
        return self.processor.apply_ducking(signal, trigger, profile)
    
    def _create_rhythmic_trigger(self, length: int, frequency: float) -> np.ndarray:
        """Create rhythmic trigger signal"""
        period_samples = int(self.sample_rate / frequency)
        trigger = np.zeros(length)
        
        # Create kicks
        kick_duration = int(0.05 * self.sample_rate)  # 50ms kick
        
        for i in range(0, length, period_samples):
            end = min(i + kick_duration, length)
            trigger[i:end] = 1.0
        
        return trigger
    
    def analyze_ducking_requirements(
        self,
        tracks: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], str]:
        """Analyze tracks and suggest ducking relationships"""
        suggestions = {}
        
        # Detect track types
        track_types = {}
        for name, audio in tracks.items():
            if "dialogue" in name.lower() or "voice" in name.lower():
                track_types[name] = "dialogue"
            elif "music" in name.lower():
                track_types[name] = "music"
            elif "sfx" in name.lower() or "effect" in name.lower():
                track_types[name] = "effects"
            else:
                # Analyze content
                freq_content = self.processor.analyzer.analyze_frequency_content(audio)
                if freq_content["mid_energy"] > freq_content["bass_energy"]:
                    track_types[name] = "dialogue"
                else:
                    track_types[name] = "music"
        
        # Create ducking matrix
        for trigger_name, trigger_type in track_types.items():
            for target_name, target_type in track_types.items():
                if trigger_name == target_name:
                    continue
                
                # Dialogue ducks everything
                if trigger_type == "dialogue" and target_type in ["music", "effects"]:
                    suggestions[(trigger_name, target_name)] = "moderate"
                
                # Effects duck music
                elif trigger_type == "effects" and target_type == "music":
                    suggestions[(trigger_name, target_name)] = "gentle"
        
        return suggestions

def db_to_linear(db: float) -> float:
    """Convert dB to linear amplitude"""
    return 10 ** (db / 20)

def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to dB"""
    return 20 * np.log10(max(linear, 1e-10))

# Example usage
if __name__ == "__main__":
    # Create system
    ducking_system = AutoDuckingSystem()
    
    # Create test signals
    sample_rate = 48000
    duration = 5.0
    samples = int(duration * sample_rate)
    
    # Dialogue (sine wave bursts)
    t = np.linspace(0, duration, samples)
    dialogue = np.zeros(samples)
    for i in range(5):
        start = int(i * sample_rate)
        end = int((i + 0.5) * sample_rate)
        dialogue[start:end] = np.sin(2 * np.pi * 440 * t[start:end])
    
    # Music (continuous tone)
    music = 0.5 * np.sin(2 * np.pi * 220 * t)
    
    # Apply auto-ducking
    result = ducking_system.auto_duck_dialogue(dialogue, music)
    
    print(f"✅ Audio ducking applied")
    print(f"  Original music level: {np.sqrt(np.mean(music**2)):.3f}")
    print(f"  Ducked music level: {np.sqrt(np.mean(result['music']**2)):.3f}")