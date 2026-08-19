# Hosted TRIBE v2 endpoint

This is a private GPU FastAPI container for a Hugging Face Inference Endpoint.
It accepts only a stimulus video and returns a `.npy` float32 cortical
prediction. The NeuroTRIBE app keeps the observed fMRI, comparison, and report
on the local machine.

## Publish the image

Use a Docker Hub, Amazon ECR, Azure ACR, or Google Artifact Registry image that
Hugging Face can pull. Replace `YOUR_REGISTRY` below:

```powershell
docker build --platform linux/amd64 -t YOUR_REGISTRY/neurotribe-hf:v1 deploy/hf-tribe-endpoint
docker push YOUR_REGISTRY/neurotribe-hf:v1
```

## Create the private endpoint

In Hugging Face Inference Endpoints, create a **Custom container** endpoint:

- Model repository: `facebook/tribev2`
- Container image: `YOUR_REGISTRY/neurotribe-hf:v1`
- Health route: `/health`
- GPU: a single 24 GB NVIDIA GPU (A10G or L4 minimum)
- Endpoint visibility: **Protected**

Wait for its health check to become `Running`, then copy the HTTPS endpoint
URL. Create an HF access token with permission to call that protected endpoint.

## Connect the local app

Add these values to the root `.env` file, then run `docker compose up -d --build api`:

```dotenv
NEUROTRIBE__TRIBE__HOSTED__ENDPOINT_URL=https://YOUR-ENDPOINT.endpoints.huggingface.cloud
HF_TRIBE_ENDPOINT_TOKEN=hf_your_token
```

The endpoint download and first GPU model load can take several minutes. Later
requests reuse the managed endpoint's model cache. Do not put the token in a
browser variable or commit it to source control.
