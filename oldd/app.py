import os
import torch
import torch.nn.functional as F
import torchaudio
import yaml
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from huggingface_hub import hf_hub_download
from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.campplus.DTDNN import CAMPPlus

class VoiceConverter:
    def __init__(self, checkpoint_path=None, config_path=None, device=None, fp16=True):
        """
        Initialize the voice converter with model paths and configurations
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config
            device: torch device (cuda/cpu)
            fp16: Whether to use half precision
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        
        # Load models and configurations
        self.load_models(checkpoint_path, config_path)
        
        # Initialize state
        self.prompt_condition = None
        self.mel2 = None
        self.style2 = None
        self.reference_wav_name = ""

    def load_from_hf(self):
        """Load model files from HuggingFace"""
        try:
            # Download checkpoint from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id="Plachta/Seed-VC",
                filename="DiT_uvit_tat_xlsr_ema.pth"
            )
            
            # Download config from HuggingFace
            config_path = hf_hub_download(
                repo_id="Plachta/Seed-VC",
                filename="config_dit_mel_seed_uvit_xlsr_tiny.yml"
            )
            
            # Download CAMP model
            camp_path = hf_hub_download(
                repo_id="funasr/campplus",
                filename="campplus_cn_common.bin"
            )
            
            self.camp_path = camp_path
            return checkpoint_path, config_path
            
        except Exception as e:
            raise Exception(f"Failed to load models from HuggingFace: {str(e)}")
        
    def load_models(self, checkpoint_path, config_path):
        """Load all required models and configurations"""
        # Load from HuggingFace if paths not provided
        if checkpoint_path is None or config_path is None:
            checkpoint_path, config_path = self.load_from_hf()
            
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        
        # Build main model
        self.model = build_model(model_params, stage="DiT")
        
        # Load checkpoint
        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        
        # Move model to device
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
            
        # Setup model caches
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Initialize CAMP model
        self.camp_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.camp_model.load_state_dict(
            torch.load(self.camp_path, map_location="cpu")
        )
        self.camp_model.eval()
        self.camp_model.to(self.device)
        
        # Initialize speech tokenizer
        self._init_speech_tokenizer(model_params.speech_tokenizer)
        
        # Initialize vocoder
        self._init_vocoder(model_params.vocoder)
        
        # Store audio parameters
        self.sr = config["preprocess_params"]["sr"]
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        
    def _init_speech_tokenizer(self, tokenizer_params):
        """Initialize the speech tokenizer"""
        if tokenizer_params.type == "xlsr":
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                tokenizer_params.name
            )
            model = Wav2Vec2Model.from_pretrained(tokenizer_params.name)
            
            # Trim layers if specified
            if hasattr(tokenizer_params, 'output_layer'):
                model.encoder.layers = model.encoder.layers[:tokenizer_params.output_layer]
                
            model = model.to(self.device)
            model.eval()
            
            if self.fp16:
                model = model.half()
                
            self.speech_tokenizer = model
            self.feature_extractor = feature_extractor
            
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_params.type}")
            
    def _init_vocoder(self, vocoder_params):
        """Initialize the vocoder"""
        if vocoder_params.type == "bigvgan":
            from modules.bigvgan import bigvgan
            self.vocoder = bigvgan.BigVGAN.from_pretrained(
                vocoder_params.name,
                use_cuda_kernel=False
            )
            self.vocoder.remove_weight_norm()
            self.vocoder.eval().to(self.device)
            
        elif vocoder_params.type == "hifigan":
            # Initialize HiFiGAN vocoder
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            
            with open('configs/hifigan.yml', 'r') as f:
                hift_config = yaml.safe_load(f)
                
            self.vocoder = HiFTGenerator(
                **hift_config['hift'],
                f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor'])
            )
            
            hift_path = hf_hub_download(
                "FunAudioLLM/CosyVoice-300M",
                'hift.pt'
            )
            
            self.vocoder.load_state_dict(torch.load(hift_path, map_location='cpu'))
            self.vocoder.eval().to(self.device)
            
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_params.type}")
    
    def convert_voice(self, reference_wav_path, input_wav_path, 
                     diffusion_steps=10, inference_cfg_rate=0.7):
        """
        Convert voice from input audio to match reference audio
        
        Args:
            reference_wav_path: Path to reference audio file
            input_wav_path: Path to input audio file
            diffusion_steps: Number of diffusion steps
            inference_cfg_rate: Inference configuration rate
            
        Returns:
            torch.Tensor: Converted audio waveform
        """
        # Load audio files
        reference_wav, _ = librosa.load(reference_wav_path, sr=self.sr)
        input_wav, _ = librosa.load(input_wav_path, sr=self.sr)
        
        # Convert to tensor
        reference_wav = torch.from_numpy(reference_wav).to(self.device)
        input_wav = torch.from_numpy(input_wav).to(self.device)
        
        # Process reference audio
        if self.prompt_condition is None or self.reference_wav_name != reference_wav_path:
            self._process_reference(reference_wav, reference_wav_path)
            
        # Convert input audio to 16kHz for processing
        input_wav_16k = torchaudio.functional.resample(input_wav, self.sr, 16000)
        
        # Generate semantic tokens
        with torch.cuda.amp.autocast(enabled=self.fp16):
            semantic_tokens = self._extract_semantic_tokens(input_wav_16k)
            
        # Prepare for inference
        target_lengths = torch.LongTensor([semantic_tokens.size(1)]).to(self.device)
        
        # Generate converted audio
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                converted_mel = self.model.cfm.inference(
                    semantic_tokens,
                    target_lengths,
                    self.mel2,
                    self.style2,
                    None,
                    n_timesteps=diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                
                converted_audio = self.vocoder(converted_mel).squeeze()
                
        return converted_audio
    
    def _extract_semantic_tokens(self, waves_16k):
        """Extract semantic tokens from audio using the speech tokenizer"""
        waves_list = [waves_16k[i].cpu().numpy() for i in range(len(waves_16k))]
        inputs = self.feature_extractor(
            waves_list,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            sampling_rate=16000
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.speech_tokenizer(inputs.input_values.half())
            
        return outputs.last_hidden_state.float()
    
    def _process_reference(self, reference_wav, reference_wav_path):
        """Process reference audio to extract style and mel features"""
        # Resample to 16kHz for speech encoder
        waves_16k = torchaudio.functional.resample(reference_wav, self.sr, 16000)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_tokens(waves_16k.unsqueeze(0))
        
        # Extract style embedding
        mel_features = torchaudio.compliance.kaldi.fbank(
            waves_16k.unsqueeze(0), 
            num_mel_bins=80,
            sample_frequency=16000
        )
        mel_features = mel_features - mel_features.mean(dim=0, keepdim=True)
        self.style2 = self.camp_model(mel_features.unsqueeze(0))
        
        # Generate mel spectrogram
        self.mel2 = self._to_mel(reference_wav.unsqueeze(0))
        
        # Store reference info
        target_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.device)
        self.prompt_condition = self.model.length_regulator(
            semantic_features, 
            ylens=target_lengths,
            n_quantizers=3,
            f0=None
        )[0]
        self.reference_wav_name = reference_wav_path
        
    def _to_mel(self, audio):
        """Convert audio to mel spectrogram"""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=2048,
            hop_length=self.hop_length,
            n_mels=80
        )(audio)
        
    def save_audio(self, audio_tensor, output_path, sample_rate=None):
        """Save converted audio to file"""
        if sample_rate is None:
            sample_rate = self.sr
        torchaudio.save(output_path, audio_tensor.cpu().unsqueeze(0), sample_rate)

# Example usage:
if __name__ == "__main__":
    # Initialize converter
    converter = VoiceConverter()
    
    # Convert voice
    converted_audio = converter.convert_voice(
        reference_wav_path="examples/reference/s1p1.wav",
        input_wav_path="examples/source/jay_0.wav",
        diffusion_steps=10,
        inference_cfg_rate=0.7
    )
    
    # Save converted audio
    converter.save_audio(converted_audio, "output.wav")