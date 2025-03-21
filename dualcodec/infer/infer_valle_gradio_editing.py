import json
import re
import tempfile
from collections import OrderedDict

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
import hydra

from loguru import logger

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


# from dualcodec.model_tts.valle_ar import ValleARInference

def load_dualcodec_valle_12hzv1():
    TTS_MODEL_CFG = {
        "model": "valle_ar",
        "ckpt_path": "dualcodec_tts_ckpts/dualcodec_valle_ar_12hzv1.safetensors",
        "cfg_path": "conf_tts/model/valle_ar/llama_250M.yaml"
    }
    from dualcodec.infer.utils_infer import load_checkpoint
    model_cfg_path = TTS_MODEL_CFG["cfg_path"]
    # instantiate model
    with hydra.initialize(config_path=model_cfg_path):
        cfg = hydra.compose(config_name=model_cfg_path)
    model = hydra.utils.instantiate(cfg.model)
    ckpt_path = TTS_MODEL_CFG["ckpt_path"]
    load_checkpoint(model, ckpt_path)
    return model

if __name__ == "__main__":
    load_dualcodec_valle_12hzv1()