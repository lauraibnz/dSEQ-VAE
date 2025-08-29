import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import librosa

from src.datasets.constants.slakh import *
from src.datasets.preprocessor import LogCompress, TakeExp


TRANSFORM = transforms.Compose([
    MelSpectrogram(sample_rate=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL),
    LogCompress(),
])
OUTPUT_DENORM = TakeExp()
OUTPUT_ACT = nn.Identity()


class Slakh(Dataset):
    def __init__(
        self,
        path_to_data: str,
        split: str = 'train',
        validation_size: float = 0.1,
        random_seed: int = 42,
        silence_threshold_db: float = -40.0  # Simple RMS threshold for music
    ):
        path_to_data = Path(path_to_data)
        assert path_to_data.exists(), f"{path_to_data} does not exist!"
        
        self.split = split
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.silence_threshold_db = silence_threshold_db
        
        # Get all track folders (handle both track_0000 and Track00001 naming conventions)
        all_tracks = []
        for subfolder in os.listdir(path_to_data):
            folder_path = path_to_data / subfolder
            if folder_path.is_dir() and (subfolder.startswith('track_') or subfolder.startswith('Track')):
                all_tracks.append(folder_path)
        
        # Split tracks first (not data)
        if split in ["train", "val"]:
            train_tracks, val_tracks = train_test_split(
                all_tracks,
                test_size=validation_size,
                random_state=random_seed
            )
            
            if split == "train":
                self.tracks = train_tracks
            elif split == "val":
                self.tracks = val_tracks
        else:
            self.tracks = all_tracks
        
        # Load and filter data
        self.audio_files = []
        self.audio_path = []
        self.labels = []
        
        print(f"Loading Slakh dataset for split: {split} ({len(self.tracks)} tracks)")
        print(f"Creating {SEQ_LEN} time-step chunks (4 seconds) with RMS > {silence_threshold_db} dB")
        self._load_data()
        
        print(f"Loaded {len(self.audio_files)} samples for {split} split")
    
    def _is_silent_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Simple silence detection using RMS energy"""
        # Convert to float32 and handle stereo
        audio_chunk = audio_chunk.astype(np.float32, copy=False)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)  # Average across channels
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_chunk * audio_chunk))
        
        # Handle true silence
        if rms == 0:
            return True
        
        # Convert to dB
        dbfs = 20 * np.log10(rms + 1e-12)
        
        return dbfs < self.silence_threshold_db
    
    def _load_data(self):
        """Load data from raw Slakh folder structure"""
        total_chunks = 0
        filtered_chunks = 0
        
        for track_folder in tqdm(self.tracks, desc="Loading tracks"):
            try:
                meta_path = track_folder / "metadata.yaml"
                if not meta_path.exists():
                    continue
                
                with open(meta_path, 'r') as file:
                    metadata = yaml.safe_load(file)
                
                stems_dir = track_folder / "stems"
                if not stems_dir.exists():
                    continue
                
                # Process each stem
                for stem_key, stem_info in metadata.get("stems", {}).items():
                    inst_class = stem_info.get("inst_class", "")
                    
                    # Skip banned instrument classes
                    if inst_class in BAN_LIST:
                        continue
                    
                    # Get instrument label
                    if inst_class in DICT_INST_TO_IDX:
                        instrument_label = DICT_INST_TO_IDX[inst_class]
                    else:
                        # Skip unknown instruments
                        continue
                    
                    # Load audio file
                    audio_path = stems_dir / f"{stem_key}.flac"
                    if not audio_path.exists():
                        continue
                    
                    try:
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(str(audio_path))
                        
                        # Resample if necessary
                        if sample_rate != SR:
                            resampler = torchaudio.transforms.Resample(sample_rate, SR)
                            waveform = resampler(waveform)
                        
                        # Convert to mono
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0)
                        
                        # Normalize
                        norm_factor = waveform.abs().max()
                        if norm_factor > 0:
                            waveform /= norm_factor
                        
                        # Segment audio into 4-second chunks (like URMP)
                        chunk_length = int(4 * SR)  # 4 seconds * 16000 Hz = 64000 samples
                        
                        # Split waveform into chunks
                        chunks = []
                        waveform_length = waveform.shape[-1]
                        for i in range(0, waveform_length, chunk_length):
                            chunk = waveform[:, i:i + chunk_length]
                            if chunk.shape[-1] >= chunk_length:  # Only use full chunks
                                chunks.append(chunk)
                        
                        # Process each chunk and filter silent ones
                        for chunk in chunks:
                            total_chunks += 1
                            
                            # Convert to numpy for RMS calculation
                            chunk_np = chunk.squeeze().numpy()
                            
                            # Check if chunk is silent using simple RMS threshold
                            if self._is_silent_chunk(chunk_np):
                                filtered_chunks += 1
                                continue
                            
                            # Apply mel-spectrogram transform
                            mel_spec = TRANSFORM(chunk.unsqueeze(0)).squeeze()  # [n_mels, T]
                            
                            self.audio_files.append(mel_spec)
                            self.audio_path.append(audio_path)
                            self.labels.append(np.array([instrument_label]))
                        
                    except Exception as e:
                        print(f"Error loading audio {audio_path}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing track {track_folder}: {e}")
                continue
        
        print(f"Filtered out {filtered_chunks}/{total_chunks} silent chunks ({filtered_chunks/total_chunks*100:.1f}%)")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        x = self.audio_files[idx]
        y = self.labels[idx]
        return idx, x.transpose(0, -1), x.size(-1), y  # Convert [features, time] to [time, features]
    
    def __repr__(self):
        return f"{self.__class__.__name__} loaded with {len(self)} samples for {self.split} split" 