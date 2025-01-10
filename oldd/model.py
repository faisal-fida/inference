import torch
import torchaudio
import numpy as np
import yaml
import os
import librosa

from modules.hf_utils import load_custom_model_from_hf
from modules.commons import recursive_munch, build_model, load_checkpoint
from modules.DTDNN import CAMPPlus
from modules.audio import mel_spectrogram


class ModelProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = True if self.device.type == "cuda" else False

        (
            self.model,
            self.semantic_fn,
            self.f0_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.to_mel,
            self.mel_fn_args,
            self.prompt_condition,
            self.mel2,
            self.style2,
        ) = self.load_models()

        self.model.eval()

        # Assume a sample rate of 16000 Hz for simplicity
        self.sample_rate = self.mel_fn_args["sampling_rate"]  # Should be 24000 as per your model
        self.sr = self.sample_rate

        # States to handle overlapping windows
        self.previous_chunk = None
        self.overlap_length = 512  # Number of samples to overlap

        # Buffer for storing incoming audio chunks
        self.audio_buffer = torch.zeros(0).to(self.device)

        # Reference audio loading
        self.ref_audio = None
        self.load_reference_audio("./reference_audio.wav")  # Path to your reference audio file

    def load_reference_audio(self, ref_audio_path):
        # Load the reference audio (target voice)
        ref_audio, _ = librosa.load(ref_audio_path, sr=self.sr)
        ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(self.device)
        self.ref_audio = ref_audio

        # Process reference audio for prompt_condition, mel2, and style2
        # Preprocess reference audio
        ori_waves_16k = torchaudio.functional.resample(self.ref_audio, self.sr, 16000)
        self.S_ori = self.semantic_fn(ori_waves_16k)

        self.mel2 = self.to_mel(self.ref_audio)
        target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        self.style2 = self.campplus_model(feat2.unsqueeze(0))

        self.prompt_condition, _, _, _, _ = self.model.length_regulator(
            self.S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )

    def load_models(self):
        # Load your model here
        # This is adapted from your provided code

        os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        )

        f0_fn = None  # Not using F0 conditioning in this example

        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        model = build_model(model_params, stage="DiT")
        hop_length = config["preprocess_params"]["spect_params"]["hop_length"]  # noqa: F841
        sr = config["preprocess_params"]["sr"]

        # Load checkpoints
        model, _, _, _ = load_checkpoint(
            model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Load additional modules
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(self.device)

        vocoder_type = model_params.vocoder.type

        if vocoder_type == "bigvgan":
            from modules.bigvgan import bigvgan

            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            # remove weight norm in the model and set to eval mode
            bigvgan_model.remove_weight_norm()
            bigvgan_model = bigvgan_model.eval().to(self.device)
            vocoder_fn = bigvgan_model
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        speech_tokenizer_type = model_params.speech_tokenizer.type
        if speech_tokenizer_type == "whisper":
            from transformers import AutoFeatureExtractor, WhisperModel

            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(
                whisper_name, torch_dtype=torch.float16
            ).to(self.device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor(
                    [waves_16k.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                ori_input_features = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(
                        ori_input_features.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
                return S_ori

        else:
            raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

        # Generate mel spectrograms
        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": sr,
            "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
            else 8000,
            "center": False,
        }

        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)  # noqa: E731

        return (
            model,
            semantic_fn,
            f0_fn,
            vocoder_fn,
            campplus_model,
            to_mel,
            mel_fn_args,
            None,  # prompt_condition placeholder
            None,  # mel2 placeholder
            None,  # style2 placeholder
        )

    def process_audio_chunk(self, audio_chunk):
        # Convert bytes to tensor
        audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)

        # Append audio to buffer
        self.audio_buffer = torch.cat([self.audio_buffer, audio_tensor], dim=1)

        # Define minimum processing length (e.g., 0.5 seconds of audio)
        min_length = int(0.5 * self.sr)
        if self.audio_buffer.shape[1] < min_length:
            # Accumulate more audio before processing
            return b""

        # Process audio in chunks
        chunk_length = min_length  # Adjust as needed

        # Extract chunk from the buffer
        input_audio = self.audio_buffer[:, :chunk_length]
        self.audio_buffer = self.audio_buffer[:, chunk_length:]

        # Resample input audio to 16kHz
        input_wav_res = torchaudio.functional.resample(input_audio, self.sr, 16000)

        # Extract semantic features
        S_alt = self.semantic_fn(input_wav_res)

        # Style transfer
        target_lengths = torch.LongTensor([S_alt.size(1)]).to(self.device)
        cond, _, _, _, _ = self.model.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=None
        )

        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

        # Voice Conversion
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32
        ):
            diffusion_steps = 30  # Adjust as needed
            inference_cfg_rate = 0.7
            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.device),
                self.mel2,
                self.style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]

        # Vocoder synthesis
        vc_wave = self.vocoder_fn(vc_target.float()).squeeze()

        output_audio = vc_wave.detach().cpu().numpy().astype(np.float32)

        # Overlapping and crossfading
        if self.previous_chunk is not None:
            overlap = self.overlap_length
            output_audio = self.crossfade(self.previous_chunk, output_audio, overlap)
            self.previous_chunk = output_audio[-overlap:]
        else:
            overlap = self.overlap_length
            self.previous_chunk = output_audio[-overlap:]

        # Convert numpy array to bytes
        output_bytes = output_audio.tobytes()

        return output_bytes

    def crossfade(self, chunk1, chunk2, overlap):
        fade_out = np.linspace(1, 0, overlap)
        fade_in = np.linspace(0, 1, overlap)
        chunk1[-overlap:] = chunk1[-overlap:] * fade_out
        chunk2[:overlap] = chunk2[:overlap] * fade_in
        crossfaded = np.concatenate(
            [chunk1[:-overlap], chunk1[-overlap:] + chunk2[:overlap], chunk2[overlap:]]
        )
        return crossfaded


# Instantiate the ModelProcessor
model_processor = ModelProcessor()
