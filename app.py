from realtime import VoiceChanger, load_models
import argparse
import torch
from modules.commons import str2bool

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint-path", type=str, default=None, help="Path to the model checkpoint"
)
parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
parser.add_argument(
    "--reference-audio-path", type=str, default="examples/reference/azuma_0.wav", help="Path to the reference audio"
)
parser.add_argument(
    "--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True
)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)

args = parser.parse_args()


cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} to Load models")
model_set = load_models(args)

config = {
    "block_time": 0.25,
    "crossfade_time": 0.05,
    "extra_time_ce": 2.5,
    "extra_time": 0.5,
    "extra_time_right": 2.0,
    "inference_cfg_rate": 0.7,
    "max_prompt_length": 3.0,
    "diffusion_steps": 10,
}

voice_changer = VoiceChanger(model_set, args.reference_audio_path, device, config)


import librosa
import torch
import soundfile as sf

def process_audio_file(file_path, processor):
    # Load the audio file
    audio_data, sr = librosa.load(file_path, sr=None)

    # Process the entire audio data as a single chunk
    processed_output = processor.process_audio_chunk(audio_data)
    print("Shape:", processed_output.shape)
    
    
process_audio_file("examples/source/source_s1.wav", voice_changer)

