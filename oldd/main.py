import os
import sys
import torch
import yaml
import librosa
import torchaudio
import torchaudio.transforms as tat
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv
import sounddevice as sd
import argparse
from modules.commons import str2bool, recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf
from funasr import AutoModel

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

# Global variables
device = None
flag_vc = False
prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds
fp16 = False

@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav_res,
                 block_frame_16k,
                 skip_head,
                 skip_tail,
                 return_length,
                 diffusion_steps,
                 inference_cfg_rate,
                 max_prompt_length,
                 cd_difference=2.0):
    global prompt_condition, mel2, style2, reference_wav_name, prompt_len, ce_dit_difference
    
    (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]

    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")

    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    converted_waves_16k = input_wav_res
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))

    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor(
        [(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]
    ).to(S_alt.device)
    
    cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target).squeeze()

    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output

def load_models(args):
    global fp16, device
    fp16 = args.fp16
    print(f"Using fp16: {fp16}")

    if args.checkpoint_path is None or args.checkpoint_path == "":
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_uvit_tat_xlsr_ema.pth",
            "config_dit_mel_seed_uvit_xlsr_tiny.yml"
        )
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    
    model, _, _, _ = load_checkpoint(
        model, None, dit_checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False
    )
    
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load CAMP+ model
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # Load vocoder
    vocoder_type = model_params.vocoder.type
    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos, _, _, _ = load_checkpoint(
            vocos, None, vocos_path,
            load_only_params=True, ignore_modules=[], is_distributed=False
        )
        _ = [vocos[key].eval().to(device) for key in vocos]
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    # Load speech tokenizer
    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features,
                attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
            
    elif speech_tokenizer_type in ['cnhubert', 'xlsr']:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel
        model_name = config['model_params']['speech_tokenizer']['name']
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        if speech_tokenizer_type == 'cnhubert':
            model = HubertModel.from_pretrained(model_name)
        else:
            model = Wav2Vec2Model.from_pretrained(model_name)
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            model.encoder.layers = model.encoder.layers[:output_layer]
            
        model = model.to(device).eval().half()

        def semantic_fn(waves_16k):
            input_values = feature_extractor(
                [waves_16k[i].cpu().numpy() for i in range(len(waves_16k))],
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000
            ).to(device)
            
            with torch.no_grad():
                outputs = model(input_values.input_values.half())
            return outputs.last_hidden_state.float()
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": config['preprocess_params']['sr'],
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args)

def process_audio(input_audio, reference_audio, model_set, config):
    """Process audio using the voice conversion model"""
    # Initialize parameters
    sr = model_set[-1]["sampling_rate"]
    zc = sr // 50
    block_frame = int(np.round(config.block_time * sr / zc)) * zc
    block_frame_16k = 320 * block_frame // zc
    
    # Prepare input
    input_wav = torch.from_numpy(input_audio).to(device)
    input_wav_res = torchaudio.functional.resample(input_wav, sr, 16000)
    
    # Process
    output = custom_infer(
        model_set,
        reference_audio,
        "reference",
        input_wav_res,
        block_frame_16k,
        config.skip_head,
        config.skip_tail,
        config.return_length,
        config.diffusion_steps,
        config.inference_cfg_rate,
        config.max_prompt_length,
        config.ce_dit_difference
    )
    
    return output.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--config-path", type=str, default=None, help="Path to the model config")
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, default=True, help="Whether to use fp16")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use")
    parser.add_argument("--input", type=str, required=True, help="Input audio file path")
    parser.add_argument("--reference", type=str, required=True, help="Reference audio file path")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--block-time", type=float, default=0.25, help="Block time in seconds")
    parser.add_argument("--diffusion-steps", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7, help="Inference cfg rate")
    parser.add_argument("--max-prompt-length", type=float, default=3.0, help="Max prompt length in seconds")
    parser.add_argument("--ce-dit-difference", type=float, default=2.0, help="Content encoder DiT difference in seconds")
    args = parser.parse_args()

    # Set up device
    global device
    cuda_target = f"cuda:{args.gpu}" if args.gpu >= 0 else "cuda"
    device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class Config:
        def __init__(self, args):
            self.block_time = args.block_time
            self.diffusion_steps = args.diffusion_steps
            self.inference_cfg_rate = args.inference_cfg_rate
            self.max_prompt_length = args.max_prompt_length
            self.ce_dit_difference = args.ce_dit_difference
            
            # Additional parameters needed for processing
            self.skip_head = int(2.5 * 50)  # 2.5 seconds * 50 frames per second
            self.skip_tail = int(0.1 * 50)  # 0.1 seconds * 50 frames per second
            self.return_length = int(0.25 * 50)  # 0.25 seconds * 50 frames per second

    # Load models
    model_set = load_models(args)
    print("Models loaded successfully")

    # Load audio files
    sr = model_set[-1]["sampling_rate"]
    input_audio, _ = librosa.load(args.input, sr=sr)
    reference_audio, _ = librosa.load(args.reference, sr=sr)
    print(f"Loaded input audio: {args.input}")
    print(f"Loaded reference audio: {args.reference}")

    # Create config
    config = Config(args)

    # Process audio
    print("Processing audio...")
    output_audio = process_audio(input_audio, reference_audio, model_set, config)

    # Save output
    import soundfile as sf
    sf.write(args.output, output_audio, sr)
    print(f"Saved output audio to: {args.output}")

if __name__ == "__main__":
    main()