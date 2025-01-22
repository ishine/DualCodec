# DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation

## About

## Installation
```bash
pip install dualcodec
```

## News
- 2025-01-17: DualCodec inference code is released!

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
### DualCodec-based TTS

## Benchmark results
### DualCodec audio quality
### DualCodec-based TTS

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
pip install dualcodec[train]
```
2. Clone this repository and `cd` to project root folder.

3. To run example training on example Emilia German data:
```bash
accelerate launch train.py --config-name=codec_train \
model=dualcodec_12hz_16384_4096_8vq \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000
```
This trains from scratch a dualcodec_12hz_16384_4096_8vq model with a training batch size of 3. (typically you need larger batch sizes)

To train a 25Hz model:
```bash
accelerate launch train.py --config-name=codec_train \
model=dualcodec_25hz_16384_1024_12vq \
trainer.batch_size=3 \
data.segment_speech.segment_length=24000

```

## Citation
