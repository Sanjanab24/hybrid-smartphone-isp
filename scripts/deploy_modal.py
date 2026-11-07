"""Deploy the app to Modal (https://modal.com) as a public web URL.

The container gets 2 CPUs and 4 GB RAM, installs requirements-app.txt and serves the same Gradio app
(app/app.py) with ONNX Runtime; only the deployed student model is shipped, not the training weights.

  pip install modal
  modal setup                              # once: log in in the browser
  modal deploy scripts/deploy_modal.py     # prints the public URL (https://<workspace>--hybrid-isp-web.modal.run)

The container sleeps after 20 minutes without visitors; the next visit wakes it in about 30-60 s.
Open the URL yourself a few minutes before judging so it is warm.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
SKIP = ["**/__pycache__", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements-app.txt"))
    .pip_install("fastapi")
    .env({"GRADIO_ANALYTICS_ENABLED": "False", "MAX_MEGAPIXELS": "2.0", "ORT_THREADS": "2"})
    .add_local_dir(str(ROOT / "isp"), remote_path="/root/isp", ignore=SKIP)
    .add_local_dir(str(ROOT / "app"), remote_path="/root/app", ignore=SKIP)
    .add_local_file(str(ROOT / "weights" / "student_fp32.onnx"), remote_path="/root/weights/student_fp32.onnx")
)

app = modal.App("hybrid-isp", image=image)


@app.function(cpu=2.0, memory=4096, max_containers=1, scaledown_window=20 * 60, timeout=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    """One container (Gradio needs sticky sessions), many concurrent requests."""
    import sys

    sys.path.insert(0, "/root")
    import gradio as gr
    from fastapi import FastAPI

    import app.app as ui  # builds the Blocks, loads the ONNX student, makes example thumbnails

    ui.demo.queue(max_size=16)
    return gr.mount_gradio_app(FastAPI(), ui.demo, path="/", theme=ui.THEME, css=ui.CSS, ssr_mode=False)
