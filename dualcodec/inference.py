import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import hydra
import os
import torch
import torchaudio
import torch.nn.functional as F

class Inference:
    def __init__(
        self, model, ckpt_path, cfg, device="cuda", **kwargs
    ) -> None:
        self.model = model
        import safetensors.torch

        self.model.to(device)
        self.model.eval()
        safetensors.torch.load_model(self.model, ckpt_path, device=device, strict=False)
        self.cfg = cfg
        self.device = device
    
    def infer(self, x_wav):
        x_wav = torch.tensor(x_wav).to(self.device)
        x_wav = x_wav.reshape(1,1,-1)
        for n_quantizers in [1,2,3,4,5,6,7,8]:
            x_out = self.model.encode(x_wav, n_quantizers=n_quantizers)[0] # [1, 8, T]
            x_out = self.model.decode(x_out)
            torchaudio.save(f'{n_quantizers}.wav', x_out.squeeze(0).cpu(), 24000)
        # breakpoint()

@hydra.main(
    version_base="1.3",
    config_path="../../../conf/",
    config_name="codec_infer.yaml",
)

def prepare_model():
    import hydra
    from hydra import initialize, initialize_config_module, initialize_config_dir, compose

    with initialize(version_base="1.3", config_path="../../../conf/"):
        cfg = compose(config_name="codec_infer.yaml", overrides=[])
        print(cfg)
    model = hydra.utils.instantiate(cfg.model.model)
    import safetensors.torch

    model.cuda()
    model.eval()
    safetensors.torch.load_model(model, cfg.ckpt_path, device='cuda', strict=False)
    return model
@torch.no_grad()
@torch.cuda.amp.autocast()
def _extract_semantic_code(self, input_features, attention_mask):
    """
        从输入特征中提取语义编码。
    该函数不需要梯度，因此被标记为@torch.no_grad().

    Args:
        input_features (torch.Tensor, shape=(B, T, C)): 输入特征，其中B是batch size，T是时间维度，C是通道维度。
        attention_mask (torch.Tensor, shape=(B, T)): 注意力掩码，其中元素为0表示对应位置的特征无效，非0表示有效。

    Returns:
        tuple (torch.Tensor, shape=(B, T)): 返回一个元组，包含语义编码和对应的量化索引（可选）。
            - semantic_code (torch.Tensor, shape=(B, T)): 语义编码，其中B是batch size，T是时间维度。
            - rep_index (Optional, torch.Tensor, shape=(B, T)): 对于每个时间步骤，如果存在对应的量化索引，则返回这些索引；否则返回None。
    """
    vq_emb = self.cfg.semantic_model["model"](
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)

    if (
        hasattr(self.cfg, "skip_semantic_normalize")
        and self.cfg.skip_semantic_normalize
    ):
        pass
    else:
        feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
            "std"
        ]
    return feat


@torch.no_grad()
def infer_with_semantic(audio, model=None, num_quantizers=8):
    audio = audio.reshape(1,1,-1).cuda()
    feature_extractor = self.cfg.feature_extractor
    inputs = feature_extractor(
        prompt_speech, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs["input_features"][0]
    attention_mask = inputs["attention_mask"][0]

    input_features = input_features.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)





    out = model(audio, )


    compressed = model.encode(audio, num_quantizers)[1]  # compressed.shape: [1, 8, T/320]

    # decode from codes
    out = model.decode(model.quantizer.from_codes(compressed)[0]).squeeze(0)  # out.shape: [1, 1, T]
    assert compressed.shape[1] == num_quantizers

    out = pad_to_length(out, audio.shape[-1])
    return out, compressed
    # The length of `compressed` might be shorter `x` due to padding.
    # To mitigate this when calculating loss, you could manually pad `compressed` with zeros to match the length of `x`.
@torch.no_grad()
def infer(audio, model=None, num_quantizers=8):
    audio = audio.reshape(1,1,-1).cuda()
    compressed = model.encode(audio, n_quantizers=num_quantizers)[1]  # compressed.shape: [1, 8, T/320]

    # decode from codes
    out = model.decoder(model.quantizer.from_codes(compressed)[0]).squeeze(0)  # out.shape: [1, 1, T]
    assert compressed.shape[1] == num_quantizers

    out = pad_to_length(out, audio.shape[-1])
    return out, compressed
#     # The length of `compressed` might be shorter `x` due to padding.
#     # To mitigate this when calculating loss, you could manually pad `compressed` with zeros to match the length of `x`.

def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded

if __name__ == "__main__":
    main(None)