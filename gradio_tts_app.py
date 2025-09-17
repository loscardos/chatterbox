import os
import time
import random
import threading
import logging
import numpy as np
import torch
import gradio as gr
from packaging import version
from chatterbox.tts import ChatterboxTTS

# --------------------
# Logging
# --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("chatterbox-tts")

# --------------------
# Config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional hard caps / queue size via env
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "128"))
MAX_WORKERS_CAP = int(os.getenv("MAX_WORKERS_CAP", "16"))  # hard ceiling
SAFETY_MARGIN = float(os.getenv("VRAM_SAFETY_MARGIN", "0.90"))  # use 90% of free VRAM

# --------------------
# Global state
# --------------------
_model = None
_model_lock = threading.Lock()

# Dynamic concurrency state
_dynamic_lock = threading.Lock()
_dynamic_workers = 1              # start conservative, expand after first measurement
_dynamic_initialized = False
_job_counter = 0

# We will increase this semaphore after we measure VRAM on first job
_gen_semaphore = threading.Semaphore(_dynamic_workers)

# Gradio needs a number at startup; we’ll bump it later to match _dynamic_workers
_default_concurrency_limit = _dynamic_workers

# --------------------
# Helpers
# --------------------
def human_mb(x_bytes: int) -> str:
    return f"{x_bytes / (1024**2):.1f} MB"

def torch_gpu_info():
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "CPU",
            "total": 0,
            "free": 0,
        }
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    name = props.name
    # mem_get_info: (free, total) in bytes
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": name,
        "total": total_b,
        "free": free_b,
    }

def update_dynamic_workers(peak_job_bytes: int):
    """
    Called once (or first successful measurement) to scale concurrency.
    """
    global _dynamic_workers, _gen_semaphore, _default_concurrency_limit, _dynamic_initialized

    if not torch.cuda.is_available():
        with _dynamic_lock:
            _dynamic_workers = 1
            _dynamic_initialized = True
        log.info("CUDA not available; concurrency fixed at 1.")
        return

    info = torch_gpu_info()
    free_b = info["free"]
    # Remove a small buffer for allocator fragmentation (~256MB)
    buffer_b = 256 * 1024 * 1024
    usable_b = max(0, int(free_b * SAFETY_MARGIN) - buffer_b)

    if peak_job_bytes <= 0:
        # Fallback if measurement failed; keep at 1
        with _dynamic_lock:
            _dynamic_workers = 1
            _dynamic_initialized = True
        log.warning("Peak VRAM per job measurement invalid; keeping workers at 1.")
        return

    # Compute potential workers
    computed = max(1, usable_b // peak_job_bytes)
    computed = int(min(computed, MAX_WORKERS_CAP))

    if computed < 1:
        computed = 1

    with _dynamic_lock:
        if computed > _dynamic_workers:
            # Increase semaphore by the delta
            delta = computed - _dynamic_workers
            for _ in range(delta):
                _gen_semaphore.release()
            _dynamic_workers = computed
            _default_concurrency_limit = computed
            _dynamic_initialized = True

    total_b = info["total"]
    log.info(
        "Dynamic concurrency set: %d workers (GPU: %s | Total %s, Free %s | "
        "Usable ~%s | Peak/job %s | Safety %.0f%% | Cap %d)",
        _dynamic_workers,
        info["name"],
        human_mb(total_b),
        human_mb(free_b),
        human_mb(usable_b),
        human_mb(peak_job_bytes),
        SAFETY_MARGIN * 100,
        MAX_WORKERS_CAP,
    )

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
                gpu = torch_gpu_info()
                if gpu["available"]:
                    log.info(
                        "Loading model on %s (Total %s, Free %s)...",
                        gpu["name"], human_mb(gpu["total"]), human_mb(gpu["free"])
                    )
                else:
                    log.info("Loading model on CPU...")
                _model = ChatterboxTTS.from_pretrained(DEVICE)
                log.info("Model loaded.")
    return _model

# --------------------
# Generation
# --------------------
def _measure_peak_and_generate(model, *args, **kwargs):
    """
    Runs one generation while measuring VRAM footprint.
    Returns (audio_sr, audio_np), peak_delta_bytes
    """
    peak_delta = 0

    # For accurate measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = torch.cuda.memory_allocated()
        before_resv  = torch.cuda.memory_reserved()

    with torch.inference_mode():
        wav = model.generate(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Some models increase "reserved" much more than "allocated"; take the larger delta
        after_alloc = torch.cuda.max_memory_allocated()
        after_resv  = torch.cuda.max_memory_reserved()
        delta_alloc = max(0, after_alloc - before_alloc)
        delta_resv  = max(0, after_resv - before_resv)
        peak_delta  = max(delta_alloc, delta_resv)

    sr = getattr(model, "sr", 22050)
    audio = wav.squeeze(0).numpy()
    return (sr, audio), peak_delta

def generate(text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    global _job_counter, _dynamic_initialized

    # Assign a job id
    with _dynamic_lock:
        _job_counter += 1
        job_id = _job_counter

    # Queue log
    log.info("Job #%d queued. Current permits: %d", job_id, _gen_semaphore._value if hasattr(_gen_semaphore, "_value") else -1)

    start_wait = time.time()
    with _gen_semaphore:
        waited = time.time() - start_wait

        # Acquire log
        gpu = torch_gpu_info()
        log.info(
            "Job #%d acquired permit after %.3fs. GPU: %s | Free %s / Total %s | Active workers ~%d",
            job_id,
            waited,
            gpu["name"],
            human_mb(gpu["free"]),
            human_mb(gpu["total"]),
            _dynamic_workers
        )

        model = get_model()

        if seed_num and int(seed_num) != 0:
            set_seed(int(seed_num))

        # First complete job: measure VRAM footprint, then scale concurrency
        if torch.cuda.is_available() and not _dynamic_initialized:
            (sr, audio), peak_b = _measure_peak_and_generate(
                model,
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            log.info(
                "Job #%d peak VRAM usage measured: %s",
                job_id, human_mb(peak_b)
            )
            update_dynamic_workers(peak_b)
        else:
            t0 = time.time()
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
            sr = getattr(model, "sr", 22050)
            audio = wav.squeeze(0).numpy()
            log.info("Job #%d generation finished in %.3fs", job_id, time.time() - t0)

        # Clean up CUDA cache between jobs to reduce fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Final log
        gpu_after = torch_gpu_info()
        log.info(
            "Job #%d done. GPU Free now %s / Total %s. Current workers ~%d",
            job_id,
            human_mb(gpu_after["free"]),
            human_mb(gpu_after["total"]),
            _dynamic_workers
        )

        return (sr, audio)

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

    # Pre-warm: show GPU info at UI load (non-blocking)
    def _startup_log():
        info = torch_gpu_info()
        if info["available"]:
            log.info(
                "Startup GPU: %s | Total %s, Free %s. Initial workers = %d (will auto-scale after first job).",
                info["name"], human_mb(info["total"]), human_mb(info["free"]), _dynamic_workers
            )
        else:
            log.info("Startup: CPU mode. Workers = 1.")
        return None

    demo.load(fn=_startup_log, inputs=[], outputs=[])

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
    # Note: default_concurrency_limit is set to 1 now; after first run we expand the semaphore,
    # and Gradio will still queue safely behind our semaphore gating.
    demo.queue(
        max_size=QUEUE_MAX_SIZE,
        default_concurrency_limit=_default_concurrency_limit
    ).launch(share=True)
