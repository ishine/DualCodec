import gradio as gr
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
from loguru import logger
import torchaudio

from dualcodec.utils.utils_infer import (
    device,
    cross_fade_duration,
    target_rms,
    nfe_step,
    speed,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from dualcodec.infer.valle.utils_valle_infer import (
    load_dualcodec_valle_ar_12hzv1,
    load_dualcodec_valle_nar_12hzv1,
    infer_process,
)
from dualcodec.utils import get_whisper_tokenizer
import dualcodec

# Load models
logger.info("Loading Valle models...")
ar_model = load_dualcodec_valle_ar_12hzv1()
nar_model = load_dualcodec_valle_nar_12hzv1()
tokenizer_model = get_whisper_tokenizer()
dualcodec_model = dualcodec.get_model("12hz_v1")
dualcodec_inference_obj = dualcodec.Inference(
    dualcodec_model=dualcodec_model, device=device, autocast=True
)
logger.info("Valle models loaded.")

def process_tts(
    ref_audio,
    ref_text,
    gen_text,
    remove_silence=False,
    cross_fade_duration=0.15,
    progress=gr.Progress(),
):
    if not ref_audio:
        gr.Warning("Please provide reference audio.")
        return None, None
    
    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return None, None

    # Preprocess reference audio and text
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)

    # Generate audio
    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ar_model_obj=ar_model,
        nar_model_obj=nar_model,
        dualcodec_inference_obj=dualcodec_inference_obj,
        tokenizer_obj=tokenizer_model,
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=gr.Info,
        progress=progress,
    )

    # Remove silence if requested
    if remove_silence and final_wave is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave) if final_wave is not None else None

# Create Gradio interface
with gr.Blocks(title="Valle TTS Demo") as demo:
    gr.Markdown("# Valle TTS Demo")
    gr.Markdown("Generate speech using reference audio and text.")
    
    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                format="wav",
            )
            ref_text = gr.Textbox(
                label="Reference Text",
                placeholder="Enter the transcript of the reference audio...",
                lines=2,
            )
            gen_text = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to generate speech for...",
                lines=4,
            )
            
            with gr.Row():
                remove_silence = gr.Checkbox(
                    label="Remove Silence",
                    value=False,
                    info="Remove long silence from generated audio",
                )
                cross_fade = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.15,
                    step=0.01,
                    label="Cross-fade Duration",
                    info="Duration of cross-fade between audio segments (seconds)",
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(
                label="Generated Audio",
                type="numpy",
                format="wav",
            )
    
    # Set up event handlers
    generate_btn.click(
        fn=process_tts,
        inputs=[
            ref_audio,
            ref_text,
            gen_text,
            remove_silence,
            cross_fade,
        ],
        outputs=[output_audio],
    )

if __name__ == "__main__":
    demo.launch(share=True) 