# Project Change History

This file documents the major milestones, architecture modifications, optimizations, and bug fixes implemented during development.

---

## 1. Dockerization & Dependency Resolution
* **Pinned HuggingFace Hub**: Added explicit pin `"huggingface_hub==0.30.2"` in the pip installation command of the `Dockerfile` to resolve compatibility issues with `transformers==4.39.2`.
* **Pytorch & MIM Setup**: Configured a reliable environment sequence in the `Dockerfile` installing CUDA 11.8 PyTorch (`torch==2.0.1+cu118`), then installing `openmim`, and installing compilation-sensitive modules (`mmcv==2.0.1`, `mmdet==3.1.0`) using MIM to avoid PyTorch library path mismatches.
* **Multi-GPU Configuration**: Prepared `docker-compose.yml` to specify GPU visibility constraints and capability allocations.

---

## 2. API Preprocessing Caching & Optimization
* **MD5 Preprocessing Caching**: Implemented a hashing function inside `api/inference_service.py` that computes the MD5 checksum of the source video/image. Pre-calculated landmarks (DWPose), face-parsing masks (FaceAlignment), VAE latents, and crop boxes are serialized and cached under `models/cache/`. Subsequent requests with the same input files bypass the heavy preprocessing stages entirely.
* **V1.5 Blending Parameters**: Exposed customizable jaw/cheek blending configurations (`extra_margin`, `parsing_mode`, `left_cheek_width`, `right_cheek_width`) in the request schemas to support MuseTalk v1.5 blending adjustments.
* **Caching Device Mismatch Resolution**: Patched the VAE latent batch loader to explicitly move cached latents to the correct device (GPU) upon loading, resolving a device mismatch crash (`Expected all tensors to be on the same device...`).

---

## 3. Real-Time Streaming & I/O Optimizations
* **FFMPEG Stdin Piping**: Rewrote the video combination loop to pipe raw BGR frame bytes directly to a running FFMPEG stdin subprocess. This avoids writing thousands of temporary PNG files to disk, saving significant storage space and accelerating compilation speed.
* **MJPEG Frame Streaming**: Added new endpoints (`/generate/stream` and `/generate/stream/json`) returning FastAPI `StreamingResponse` objects formatted as `multipart/x-mixed-replace; boundary=frame`. This yields JPEG frames in real-time as they are decoded by the VAE.

---

## 4. Default Model Configuration
* Configured MuseTalk v1.5 weights (`unet.pth` and `musetalk.json`) as the default checkpoint in `api/inference_service.py` and `app.py`.

---

## 5. Streaming Verification & Utility Integration
* **Cached Device Alignment**: Validated the device alignment fix (`latent_batch.to(device=self.device)`) for cached latents, correcting a device mismatch runtime crash inside the container.
* **Streaming Test Utility (`test_stream.py`)**: Authored a standalone validation utility that programmatically requests, decodes, and saves raw MJPEG frames (`multipart/x-mixed-replace`) in real-time, confirming stable 50 FPS generation speeds.
* **Direct Pipeline Commands**: Formulated CLI command workflows combining `curl` and `ffmpeg` to capture and extract video frames directly in the terminal during streaming.

