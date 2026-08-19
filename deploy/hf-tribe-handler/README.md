---
tags:
  - endpoints-template
  - custom-inference-handler
  - fmri
  - neuroscience
---

# Hugging Face managed TRIBE endpoint

This is the preferred deployment route for NeuroTRIBE. Hugging Face builds
this handler directly in its managed GPU environment, so no multi-gigabyte CUDA
image needs to be built or pushed from the local machine.

Publish `handler.py` and `requirements.txt` to a private Hugging Face **model**
repository. Create an Inference Endpoint from that repository using a 24 GB
NVIDIA GPU (A10G or L4 minimum) and **Protected** visibility. The inference
toolkit automatically detects `EndpointHandler`.

Configure the local app after the endpoint becomes `Running`:

```dotenv
NEUROTRIBE__TRIBE__HOSTED__ENDPOINT_URL=https://YOUR-ENDPOINT.endpoints.huggingface.cloud
NEUROTRIBE__TRIBE__HOSTED__PROTOCOL=hf-toolkit
HF_TRIBE_ENDPOINT_TOKEN=hf_your_token
```

The only request field is `inputs.video_base64`; the handler returns a mean
visual-only cortical response in `prediction_npy_base64`. Observed BOLD fMRI,
maps, deviation calculation, and report remain local.
