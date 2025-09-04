import os
import random
import threading
import numpy as np
import torch
import gradio as gr
from packaging import version
from chatterbox.tts import ChatterboxTTS

# --------------------
# Config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# How many generations can run *at the same time* on the model
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "128"))

# --------------------
# Global, shared model (one per process)
# --------------------
_model = None
_model_lock = threading.Lock()
# Semaphore to prevent too many concurrent generations on the GPU
_gen_semaphore = threading.Semaphore(MAX_WORKERS)

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ChatterboxTTS.from_pretrained(DEVICE)
    return _model

def generate(text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    # Acquire a slot to run generation (others will queue but different users can still run in parallel)
    with _gen_semaphore:
        model = get_model()

        if seed_num and int(seed_num) != 0:
            set_seed(int(seed_num))

        with torch.inference_mode():
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        return (model.sr, wav.squeeze(0).numpy())

# --------------------
# UI
# --------------------
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File",
                value=None
            )
            exaggeration = gr.Slider(0.25, 2, step=.05,
                                     label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                                     value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="min_p || Newer Sampler. Recommend 0.02 → 0.1. Handles Higher Temperatures better. 0.00 Disables",
                                  value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01,
                                  label="top_p || Original Sampler. 1.0 Disables (recommended). Original 0.8",
                                  value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1,
                                               label="repetition_penalty",
                                               value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    # Pre-warm the model once when the app starts (first visitor triggers it)
    demo.load(fn=lambda: None, inputs=[], outputs=[])

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    # Use the right queue arg for your Gradio version
    v = version.parse(gr.__version__)
    if v >= version.parse("4.0.0"):
        demo.queue(max_size=QUEUE_MAX_SIZE, concurrency_count=MAX_WORKERS).launch(share=True)
    else:
        # Older Gradio (pre-4.0) used default_concurrency_limit
        demo.queue(max_size=QUEUE_MAX_SIZE, default_concurrency_limit=MAX_WORKERS).launch(share=True)
