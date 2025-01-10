import os
import numpy as np
import torch
import yaml
import torchaudio
from collections import deque
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from modules.hifigan.generator import HiFTGenerator
from modules.hifigan.f0_predictor import ConvRNNF0Predictor
from modules.campplus.DTDNN import CAMPPlus
from modules.commons import *
from hf_utils import load_custom_model_from_hf

class StreamingVoiceConverter:
    def __init__(
        self,
        checkpoint_path: str = "DiT_uvit_tat_xlsr_ema.pth",
        config_path: str = "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.config = yaml.safe_load(open(config_path, "r"))
        self.model_params = recursive_munch(self.config["model_params"])
        self.sr = self.config["preprocess_params"]["sr"]
        self.hop_length = self.config["preprocess_params"]["spect_params"]["hop_length"]
        
        # Initialize buffers
        self.input_buffer = deque(maxlen=self.sr * 30)  # 30 seconds max buffer
        self.semantic_buffer = None
        self.output_buffer = deque(maxlen=self.sr * 30)
        
        # Load models
        self.load_models(checkpoint_path)
        
        # Processing parameters
        self.chunk_size = self.sr * 5  # Process 5 seconds at a time
        self.overlap = self.sr * 1  # 1 second overlap
        
    def load_models(self, checkpoint_path: str):
        # Load DiT model
        self.model = build_model(self.model_params, stage="DiT")
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                                         "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
        self.model, _, _, _ = load_checkpoint(
            self.model, None, dit_checkpoint_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        
        # Load XLSR model
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(
            self.config['model_params']['speech_tokenizer']['name']
        )
        output_layer = self.config['model_params']['speech_tokenizer']['output_layer']
        self.wav2vec_model.encoder.layers = self.wav2vec_model.encoder.layers[:output_layer]
        self.wav2vec_model = self.wav2vec_model.eval().to(self.device)
        
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config['model_params']['speech_tokenizer']['name']
        )
        
        # Load HiFiGAN vocoder
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        self.vocoder = HiFTGenerator(
            **hift_config['hift'], 
            f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor'])
        )
        self.vocoder.load_state_dict(torch.load('hift.pt', map_location='cpu'))
        self.vocoder.eval().to(self.device)
        
        # Load CAMPPlus model
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(
            torch.load('campplus_cn_common.bin', map_location='cpu')
        )
        self.campplus_model.eval().to(self.device)
        
        # Setup mel spectrogram converter
        self.mel_fn_args = {
            "n_fft": self.config['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.config['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.sr,
            "fmin": self.config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        
    def process_semantic_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract semantic features using XLSR"""
        with torch.no_grad():
            inputs = self.wav2vec_feature_extractor(
                audio.cpu().numpy(),
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000
            ).to(self.device)
            
            outputs = self.wav2vec_model(inputs.input_values)
            return outputs.last_hidden_state
            
    def get_speaker_embedding(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding using CAMPPlus"""
        feat = torchaudio.compliance.kaldi.fbank(
            reference_audio,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus_model(feat.unsqueeze(0))
        
    def stream_input(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process streaming input audio chunk"""
        # Convert to torch tensor and append to buffer
        audio_chunk = torch.from_numpy(audio_chunk).float()
        self.input_buffer.extend(audio_chunk)
        
        # If we have enough data, process a chunk
        if len(self.input_buffer) >= self.chunk_size:
            # Get chunk with overlap
            chunk = torch.tensor(list(self.input_buffer)[-self.chunk_size:])
            chunk = chunk.unsqueeze(0).to(self.device)
            
            # Resample to 16kHz for semantic feature extraction
            chunk_16k = torchaudio.functional.resample(chunk, self.sr, 16000)
            
            # Extract semantic features
            semantic_features = self.process_semantic_features(chunk_16k)
            
            # Apply voice conversion
            converted_mel = self.model.cfm.inference(
                semantic_features,
                torch.LongTensor([semantic_features.size(1)]).to(self.device),
                self.reference_mel,
                self.reference_style,
                None,
                diffusion_steps=30,
                inference_cfg_rate=0.7
            )
            
            # Generate audio
            with torch.no_grad():
                converted_audio = self.vocoder(converted_mel).squeeze().cpu().numpy()
            
            # Add to output buffer with crossfade if needed
            if len(self.output_buffer) > 0:
                # Apply crossfade
                fade_len = self.overlap
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                
                overlap_start = converted_audio[:fade_len]
                previous_end = np.array(list(self.output_buffer)[-fade_len:])
                
                crossfaded = (overlap_start * fade_in) + (previous_end * fade_out)
                converted_audio[:fade_len] = crossfaded
            
            # Add to output buffer
            self.output_buffer.extend(converted_audio)
            
            # Return non-overlapping portion
            output_length = len(converted_audio) - self.overlap
            if output_length > 0:
                return np.array(list(self.output_buffer)[:output_length])
            
        return None
    
    def set_reference(self, reference_path: str, max_duration: float = 30.0):
        """Set reference speaker audio from a wav/flac file
        
        Args:
            reference_path: Path to reference audio file (.wav or .flac)
            max_duration: Maximum duration in seconds to use from reference file
        """
        # Load and trim reference audio
        audio, file_sr = torchaudio.load(reference_path)
        
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample to model sample rate if needed
        if file_sr != self.sr:
            audio = torchaudio.functional.resample(audio, file_sr, self.sr)
            
        # Trim to max duration
        max_samples = int(max_duration * self.sr)
        if audio.size(1) > max_samples:
            audio = audio[:, :max_samples]
            
        # Move to device
        reference = audio.to(self.device)
        
        # Resample to 16kHz for feature extraction
        reference_16k = torchaudio.functional.resample(reference, self.sr, 16000)
        
        # Get reference mel spectrogram
        self.reference_mel = mel_spectrogram(reference, **self.mel_fn_args)
        
        # Get reference speaker embedding
        self.reference_style = self.get_speaker_embedding(reference_16k)
        
        print(f"Loaded reference audio from {reference_path}")
        print(f"Duration: {audio.size(1)/self.sr:.2f} seconds")
        
    def reset_buffers(self):
        """Clear all internal buffers"""
        self.input_buffer.clear()
        self.semantic_buffer = None
        self.output_buffer.clear()



# Initialize
converter = StreamingVoiceConverter()

converter.set_reference("examples/reference/s1p1.wav")  # or .flac

