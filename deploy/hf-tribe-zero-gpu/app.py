"""Free on-demand visual-only TRIBE v2 inference for a Hugging Face ZeroGPU Space."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import spaces
from neuralset.events.utils import standardize_events
from tribev2 import TribeModel

# ZeroGPU's CUDA emulation makes module-level placement possible before a GPU is
# allocated. This prevents model transfer from consuming the short request slot.
MODEL = TribeModel.from_pretrained(
    "facebook/tribev2",
    device="cuda",
    config_update={
        "data.video_feature.use_audio": False,
        "data.features_to_use": ["video"],
    },
)


@spaces.GPU(duration=120)
def predict(video_path: str | None) -> str:
    """Predict one mean fsaverage5 cortical map from a short, silent video."""
    if not video_path:
        raise gr.Error("Upload a short MP4 stimulus video.")
    source = Path(video_path)
    if not source.is_file():
        raise gr.Error("The uploaded video could not be read.")
    events = standardize_events(pd.DataFrame([{
        "type": "Video", "filepath": str(source), "start": 0,
        "timeline": "request", "subject": "default",
    }]))
    prediction, _segments = MODEL.predict(events=events, verbose=False)
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[0] == 0:
        raise gr.Error("TRIBE returned no usable visual prediction.")
    output = tempfile.NamedTemporaryFile(prefix="tribe-prediction-", suffix=".npy", delete=False)
    output.close()
    np.save(output.name, prediction.mean(axis=0).astype(np.float32), allow_pickle=False)
    return output.name


with gr.Blocks(title="NeuroTRIBE free visual inference") as demo:
    gr.Markdown(
        "# NeuroTRIBE free visual inference\n"
        "This on-demand demo accepts only a short stimulus video and returns a "
        "visual-only TRIBE v2 cortical prediction. It is quota-limited; use it "
        "for a small number of exploratory runs."
    )
    video = gr.Video(label="Stimulus video", format="mp4")
    run = gr.Button("Run visual TRIBE", variant="primary")
    result = gr.File(label="TRIBE cortical prediction (.npy)")
    run.click(predict, inputs=video, outputs=result, api_name="predict")


if __name__ == "__main__":
    demo.launch()
