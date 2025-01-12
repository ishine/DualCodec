# DualCodec: A Speech Generation-Oriented Neural Audio Codec with Dual Encoding of Waveform and Self-Supervised Feature



## Installation
```bash
pip install dualcodec
```

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
huggingface-cli download amphion/dualcodec --local-dir dualcodec_ckpts
```

### 2. To inference an audio in a python script: 
```python
import dualcodec

w2v_path = "./w2v-bert-2.0" # your downloaded path
dualcodec_model_path = "./dualcodec_ckpts" # your downloaded path
model_id = "12hz_v1" # or "25hz_v1"

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
### Benchmarking

### Link to DualCodec-based TTS repositories

## Training DualCodec
Stay tuned for the training code release! Should be within two weeks.

## Citation