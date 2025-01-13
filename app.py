import gradio as gr
import torch
import torchaudio
import dualcodec

# Model configuration
MODEL_CONFIGS = {
    "12hz_v1": {"max_quantizers": 8},
    "25hz_v1": {"max_quantizers": 12}
}

w2v_path = "./w2v-bert-2.0"
dualcodec_model_path = "./dualcodec_ckpts"

# Global model variables
current_model = None
current_inference = None

def load_model(model_id):
    global current_model, current_inference
    current_model = dualcodec.get_model(model_id, dualcodec_model_path)
    current_inference = dualcodec.Inference(
        dualcodec_model=current_model,
        dualcodec_path=dualcodec_model_path,
        w2v_path=w2v_path,
        device="cpu"
    )
    return MODEL_CONFIGS[model_id]["max_quantizers"]

def process_audio(audio_file, model_id, n_quantizers, pinned):
    if current_model is None or current_inference is None:
        load_model(model_id)
        
    # Load and process audio
    audio, sr = torchaudio.load(audio_file)
    audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.reshape(1, 1, -1)
    
    # Encode and decode
    semantic_codes, acoustic_codes = current_inference.encode(audio, n_quantizers=n_quantizers)
    out_audio = current_model.decode_from_codes(semantic_codes, acoustic_codes)
    
    # Prepare outputs
    generated_audio = (24000, out_audio.cpu().numpy().squeeze())
    
    # Update pinned state
    pinned = {
        "audio": generated_audio,
        "metadata": f"Model: {model_id}, VQs: {n_quantizers}"
    }
    return (24000, audio.cpu().numpy().squeeze()), generated_audio, pinned

def update_slider(model_id):
    return gr.update(maximum=MODEL_CONFIGS[model_id]["max_quantizers"])

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# DualCodec Audio Demo")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value="12hz_v1",
            label="Model"
        )
        n_quantizers = gr.Slider(
            minimum=1,
            maximum=MODEL_CONFIGS["12hz_v1"]["max_quantizers"],
            step=1,
            value=8,
            label="Number of Quantizers"
        )
    
    audio_input = gr.Audio(type="filepath", label="Input Audio")
    inference_button = gr.Button("Run Inference")
    
    with gr.Row():
        audio_output_orig = gr.Audio(label="Original Audio")
        audio_output_recon = gr.Audio(label="Reconstructed Audio")
    
    # Pinned card
    with gr.Row():
        pinned_audio = gr.Audio(label="Pinned Audio")
        pinned_metadata = gr.Textbox(label="Pinned Metadata", interactive=False)
    
    # State to store pinned audio
    pinned_state = gr.State({"audio": None, "metadata": ""})
    
    # Set up interactions
    model_dropdown.change(fn=update_slider, inputs=model_dropdown, outputs=n_quantizers)
    inference_button.click(
        fn=process_audio,
        inputs=[audio_input, model_dropdown, n_quantizers, pinned_state],
        outputs=[audio_output_orig, audio_output_recon, pinned_state]
    )
    pinned_state.change(
        fn=lambda pinned: (pinned["audio"], pinned["metadata"]),
        inputs=pinned_state,
        outputs=[pinned_audio, pinned_metadata]
    )

if __name__ == "__main__":
    demo.launch()
