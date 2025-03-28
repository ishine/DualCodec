import torch
from cached_path import cached_path
from functools import partial

def load_voicebox_300M_model():
    TTS_MODEL_CFG = {
        "model": "voicebox_300M",
        # "ckpt_path": "hf://amphion/dualcodec-tts/dualcodec_valle_ar_12hzv1.safetensors",
        "ckpt_path": "/gluster-ssd-tts/tts_share_training_logs/lijiaqi18/job-0bbab25ef77fae6c/voicebox_train/checkpoint/epoch-0002_step-0250000_loss-0.335350-voicebox_train/model.safetensors",
        # "cfg_path": "../../conf/model/valle_ar/llama_250M.yaml"
    }
    from dualcodec.model_tts.voicebox.voicebox_models import voicebox_300M
    model = voicebox_300M()
    # load model
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    ckpt_path = cached_path(ckpt_path)
    import safetensors.torch
    model = safetensors.torch.load_model(model, ckpt_path)
    return model

def load_dualcodec_12hzv1_model():
    import dualcodec
    dualcodec_model = dualcodec.get_model("12hz_v1")
    dualcodec_inference_obj = dualcodec.Inference(dualcodec_model=dualcodec_model, device=device, autocast=True)
    return dualcodec_inference_obj

def get_vocoder_decode_func_and_mel_spec():
    from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos
    vocos_model, mel_model = get_vocos_model_spectrogram()
    infer_vocos = partial(mel_to_wav_vocos, vocos_model)
    return infer_vocos, mel_model

@torch.inference_mode()
def voicebox_inference(
    voicebox_model_obj,
    vocoder_decode_func,
    mel_spec_extractor_func,
    combine_semantic_code, # shape [b t]
    prompt_acoustic_code, # shape [1, q, t]
    device='cuda',
):
    def code2mel(self, combine_semantic_code: torch.Tensor, prompt_speech):
        cond_feature = voicebox_model_obj.cond_emb(combine_semantic_code)
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=voicebox_model_obj.cond_scale_factor,
        ).transpose(1, 2)

        if prompt_speech is not None:
            prompt_mel_feat = mel_spec_extractor_func(
                torch.tensor(prompt_speech).unsqueeze(0), 
                device=device,
            )
        else:
            prompt_mel_feat = None

        predict_mel = voicebox_model_obj.reverse_diffusion(
            cond_feature,
            prompt_mel_feat,
            n_timesteps=32,
            cfg=2.0,
            rescale_cfg=0.75,
        )

        return predict_mel

    predicted_mel = code2mel(
        voicebox_model_obj,
        combine_semantic_code,
        prompt_acoustic_code,
    ) # [b, 1, t]

    predicted_audio = vocoder_decode_func(predicted_mel)
    return predicted_audio

if __name__ == '__main__':
    from dualcodec.model_tts.voicebox.voicebox_models import voicebox_300M
    voicebox_model_obj = voicebox_300M()
    vocoder_decode_func = get_vocoder_decode_func_and_mel_spec()
    predicted = voicebox_inference(
        voicebox_model_obj=voicebox_model_obj,
        vocoder_decode_func=vocoder_decode_func,
        mel_spec_extractor_func=
    )