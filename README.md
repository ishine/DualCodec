# DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-brightgreen.svg?style=flat-square)](https://arxiv.org/)
[![githubio](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://dualcodec.github.io/)
[![PyPI](https://img.shields.io/pypi/v/dualcodec?color=blue&label=PyPI&logo=PyPI&style=flat-square)](https://pypi.org/project/dualcodec/)
![Amphion](https://img.shields.io/badge/Amphion-Stable_Release-blue?style=flat-square)
![GitHub](https://img.shields.io/badge/Github-Dev_Release-pink?logo=Github&style=flat-square)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VvUhsDffLdY5TdNuaqlLnYzIoXhvI8MK#scrollTo=Lsos3BK4J-4E)

## About
DualCodec is a low-frame-rate (12.5Hz or 25Hz), semantically-enhanced (with SSL feature) Neural Audio Codec designed to extract discrete tokens for efficient speech generation.

You can check out our paper and our [demo page](https://dualcodec.github.io/).
The overview of DualCodec system is shown in the following figure:

<!-- show dualcodec.png -->
![DualCodec](dualcodec.png)


## Installation
```bash
pip install dualcodec
```

## News
- 2025-01-22: I added training and finetuning instructions for DualCodec, version is v0.3.0.
- 2025-01-16: Finished releasing DualCodec inference codes, the version is v0.1.0. Latest versions are synced to pypi.

## Available models
<!-- - 12hz_v1: DualCodec model trained with 12Hz sampling rate. 
- 25hz_v1: DualCodec model trained with 25Hz sampling rate. -->

| Model_ID   | Frame Rate | RVQ Quantizers | Semantic Codebook Size (RVQ-1 Size) | Acoustic Codebook Size (RVQ-rest Size) | Training Data       |
|-----------|------------|----------------------|-------------------------------------|----------------------------------------|---------------------|
| 12hz_v1   | 12.5Hz     | Any from 1-8 (maximum 8)        | 16384                               | 4096                                   | 100K hours Emilia  |
| 25hz_v1   | 25Hz       | Any from 1-12 (maximum 12)       | 16384                               | 1024                                   | 100K hours Emilia  |


## How to inference DualCodec

### 1. Download checkpoints to local: 
```
# export HF_ENDPOINT=https://hf-mirror.com      # uncomment this to use huggingface mirror if you're in China
huggingface-cli download facebook/w2v-bert-2.0 --local-dir w2v-bert-2.0
huggingface-cli download amphion/dualcodec dualcodec_12hz_16384_4096.safetensors dualcodec_25hz_16384_1024.safetensors w2vbert2_mean_var_stats_emilia.pt --local-dir dualcodec_ckpts
```
The second command downloads the two DualCodec model (12hz_v1 and 25hz_v1) checkpoints and a w2v-bert-2 mean and variance statistics to the local directory `dualcodec_ckpts`.

### 2. To inference an audio in a python script: 
```python
import dualcodec

w2v_path = "./w2v-bert-2.0" # your downloaded path
dualcodec_model_path = "./dualcodec_ckpts" # your downloaded path
model_id = "12hz_v1" # select from available Model_IDs, "12hz_v1" or "25hz_v1"

dualcodec_model = dualcodec.get_model(model_id, dualcodec_model_path)
inference = dualcodec.Inference(dualcodec_model=dualcodec_model, dualcodec_path=dualcodec_model_path, w2v_path=w2v_path, device="cuda")

# do inference for your wav
import torchaudio
audio, sr = torchaudio.load("YOUR_WAV.wav")
# resample to 24kHz
audio = torchaudio.functional.resample(audio, sr, 24000)
audio = audio.reshape(1,1,-1)
# extract codes, for example, using 8 quantizers here:
semantic_codes, acoustic_codes = inference.encode(audio, n_quantizers=8)
# semantic_codes shape: torch.Size([1, 1, T])
# acoustic_codes shape: torch.Size([1, n_quantizers-1, T])

# produce output audio
out_audio = dualcodec_model.decode_from_codes(semantic_codes, acoustic_codes)

# save output audio
torchaudio.save("out.wav", out_audio.cpu().squeeze(0), 24000)
```

See "example.ipynb" for a running example.

## DualCodec-based TTS models
We're releasing DualCodec-based TTS models. Stay tuned!

## Finetuning DualCodec
1. Install other necessary components for training:
```bash
pip install "dualcodec[train]"
```
2. Clone this repository and `cd` to project root folder.

3. Get discriminator checkpoints:
```bash
huggingface-cli download amphion/dualcodec --local-dir dualcodec_ckpts
```

4. To run example training on Emilia German data (streaming, no need to download files. Need to access Huggingface):
```bash
accelerate launch train.py --config-name=dualcodec_ft_12hzv1 \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```
This trains from scratch a 12hz_v1 model with a training batch size of 3. (typically you need larger batch sizes)

To finetune a 25Hz_V1 model:
```bash
accelerate launch train.py --config-name=dualcodec_ft_25hzv1 \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```


## Training DualCodec from scratch
1. Install other necessary components for training:
```bash
pip install "dualcodec[train]"
```
2. Clone this repository and `cd` to project root folder.

3. To run example training on example Emilia German data:
```bash
accelerate launch train.py --config-name=codec_train \
model=dualcodec_12hz_16384_4096_8vq \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```
This trains from scratch a v1_12hz model with a training batch size of 3. (typically you need larger batch sizes)

To train a v1_25Hz model:
```bash
accelerate launch train.py --config-name=codec_train \
model=dualcodec_25hz_16384_1024_12vq \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000

```

## Citation
