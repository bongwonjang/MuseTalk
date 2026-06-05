import os
import shutil
import glob
import hashlib
import subprocess
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder


class MuseTalkInference:
    def __init__(self, use_float16: bool = True, gpu_id: int = 0):
        self.use_float16 = use_float16
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.gfpgan_restorer = None

        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.weight_dtype = torch.float32
        self.audio_processor = None
        self.whisper = None

    def load_models(self) -> None:
        if self.models_loaded:
            return

        print(f"Loading models on device: {self.device}")

        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=self.device,
        )

        self.timesteps = torch.tensor([0], device=self.device)

        if self.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        self.whisper = WhisperModel.from_pretrained("./models/whisper")
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.models_loaded = True
        print("Models loaded successfully!")

    def _load_gfpgan(self) -> None:
        if self.gfpgan_restorer is not None:
            return

        from gfpgan import GFPGANer

        print("Loading GFPGAN model...")
        self.gfpgan_restorer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
        )
        print("GFPGAN loaded!")

    def _enhance_face_aligned(self, face_crop: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """
        Enhance a pre-cropped face using GFPGAN with has_aligned=True.

        This skips face detection entirely since MuseTalk already extracted the face.
        GFPGAN expects 512x512 input, so we resize, enhance, then resize back.

        Args:
            face_crop: Face crop from MuseTalk (typically 256x256)
            weight: Blending weight (0=original, 1=fully enhanced)

        Returns:
            Enhanced face crop at original resolution
        """
        if self.gfpgan_restorer is None:
            return face_crop

        original_size = (face_crop.shape[1], face_crop.shape[0])  # (w, h)

        # GFPGAN expects 512x512 for optimal quality
        face_512 = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        try:
            # has_aligned=True skips face detection - HUGE speedup!
            # paste_back=False since we're handling the blending ourselves
            _, restored_faces, _ = self.gfpgan_restorer.enhance(
                face_512,
                has_aligned=True,
                only_center_face=False,
                paste_back=False,
                weight=weight,
            )

            if restored_faces and len(restored_faces) > 0:
                enhanced_512 = restored_faces[0]
                # Resize back to original face crop size
                enhanced_crop = cv2.resize(
                    enhanced_512, original_size, interpolation=cv2.INTER_LANCZOS4
                )
                return enhanced_crop
        except Exception as e:
            print(f"GFPGAN enhancement failed: {e}")

        return face_crop

    def get_file_hash(self, file_path: str) -> str:
        """Compute the MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _generate_frames(
        self,
        audio_path: str,
        video_path: str,
        enhance: bool = False,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        fps: int = 25,
        batch_size: int = 1,
        gfpgan_weight: float = 0.5,
    ):
        """Internal generator yielding blended BGR frames in real-time."""
        if not self.models_loaded:
            self.load_models()

        # Cache directory setup inside persistent models directory
        cache_dir = "./models/cache"
        os.makedirs(cache_dir, exist_ok=True)

        file_hash = self.get_file_hash(video_path)
        cache_filename = f"{file_hash}_bbox{bbox_shift}_margin{extra_margin}_mode{parsing_mode}_l{left_cheek_width}_r{right_cheek_width}.pt"
        cache_file_path = os.path.join(cache_dir, cache_filename)

        # 1. Read frames from source (OpenCV captures BGR natively)
        if get_file_type(video_path) == "video":
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_list.append(frame)
            cap.release()
            fps = int(get_video_fps(video_path))
        elif get_file_type(video_path) == "image":
            frame_list = [cv2.imread(video_path)]
        else:
            raise ValueError(f"{video_path} should be a video file or an image file")

        # 2. Get preprocessing material (either from cache or by computing it)
        if os.path.exists(cache_file_path):
            print(f"Loading preprocessed materials from cache: {cache_file_path}")
            cache_data = torch.load(cache_file_path, map_location="cpu")
            coord_list = cache_data["coord_list"]
            input_latent_list = cache_data["input_latent_list"]
            mask_list = cache_data["mask_list"]
            mask_coords_list = cache_data["mask_coords_list"]
        else:
            print("Cache miss. Running preprocessing...")
            import tempfile
            temp_dir = tempfile.mkdtemp()
            try:
                if get_file_type(video_path) == "video":
                    # Write frames to temp files so get_landmark_and_bbox (DWPose) can read them
                    save_dir_full = os.path.join(temp_dir, "frames")
                    os.makedirs(save_dir_full, exist_ok=True)
                    for i, im in enumerate(frame_list):
                        cv2.imwrite(f"{save_dir_full}/{i:08d}.png", im)
                    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, "*.[jpJP][pnPN]*[gG]")))
                else:
                    input_img_list = [video_path]

                # Run landmarks & bbox detection
                coord_list, frame_list_processed = get_landmark_and_bbox(input_img_list, bbox_shift)
                frame_list = frame_list_processed

                fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
                input_latent_list = []
                mask_list = []
                mask_coords_list = []

                for bbox, frame in zip(coord_list, frame_list):
                    if bbox == coord_placeholder:
                        input_latent_list.append(None)
                        mask_list.append(None)
                        mask_coords_list.append(None)
                        continue
                    x1, y1, x2, y2 = bbox
                    y2 = y2 + extra_margin
                    y2 = min(y2, frame.shape[0])
                    crop_frame = frame[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latents = self.vae.get_latents_for_unet(crop_frame)
                    input_latent_list.append(latents)

                    # Pre-calculate face mask and crop box using FaceParsing
                    mask, crop_box = get_image_prepare_material(
                        frame, [x1, y1, x2, y2], fp=fp, mode=parsing_mode
                    )
                    mask_list.append(mask)
                    mask_coords_list.append(crop_box)

                # Save to persistent cache
                cache_data = {
                    "coord_list": coord_list,
                    "input_latent_list": input_latent_list,
                    "mask_list": mask_list,
                    "mask_coords_list": mask_coords_list,
                }
                torch.save(cache_data, cache_file_path)
                print(f"Preprocessed materials saved to cache: {cache_file_path}")
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        # 3. Audio processing
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        active_latents = [l for l in input_latent_list if l is not None]
        if not active_latents:
            raise ValueError("No face detected in any of the reference frames.")

        input_latent_list_cycle = active_latents + active_latents[::-1]
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        mask_list_cycle = mask_list + mask_list[::-1]
        mask_coords_list_cycle = mask_coords_list + mask_coords_list[::-1]

        # 4. UNet model inference & Blending loop (Batch-by-Batch Streaming)
        print("Starting inference and blending...")
        if enhance:
            self._load_gfpgan()

        video_num = len(whisper_chunks)
        device_str = str(self.device)
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device_str,
        )

        total = int(np.ceil(float(video_num) / batch_size))

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="Inference & Blending")):
            audio_feature_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(device=self.device, dtype=self.weight_dtype)

            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)

            # Process, blend, and yield the current batch of frames immediately
            for j, res_frame in enumerate(recon):
                frame_idx = i * batch_size + j
                if frame_idx >= video_num:
                    break

                bbox = coord_list_cycle[frame_idx % len(coord_list_cycle)]
                ori_frame = frame_list_cycle[frame_idx % len(frame_list_cycle)].copy()
                
                if bbox == coord_placeholder:
                    yield ori_frame
                    continue

                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, ori_frame.shape[0])

                face_crop = res_frame.astype(np.uint8)
                if enhance:
                    face_crop = self._enhance_face_aligned(face_crop, gfpgan_weight)

                try:
                    face_resized = cv2.resize(face_crop, (x2 - x1, y2 - y1))
                except Exception:
                    yield ori_frame
                    continue

                mask = mask_list_cycle[frame_idx % len(mask_list_cycle)]
                mask_crop_box = mask_coords_list_cycle[frame_idx % len(mask_coords_list_cycle)]

                if mask is None or mask_crop_box is None:
                    yield ori_frame
                    continue

                combine_frame = get_image_blending(
                    ori_frame, face_resized, [x1, y1, x2, y2], mask, mask_crop_box
                )
                yield combine_frame

    @torch.no_grad()
    def generate(
        self,
        audio_path: str,
        video_path: str,
        enhance: bool = False,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        fps: int = 25,
        batch_size: int = 1,
        output_name: Optional[str] = None,
        result_dir: str = "./results",
        gfpgan_weight: float = 0.5,
    ) -> str:
        os.makedirs(result_dir, exist_ok=True)

        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]

        if output_name:
            output_name = (
                os.path.splitext(output_name)[0] if output_name.endswith(".mp4") else output_name
            )
            output_vid_name = os.path.join(result_dir, f"{output_name}.mp4")
        else:
            output_vid_name = os.path.join(result_dir, f"{input_basename}_{audio_basename}.mp4")

        # Get frame generator
        frame_generator = self._generate_frames(
            audio_path=audio_path,
            video_path=video_path,
            enhance=enhance,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            fps=fps,
            batch_size=batch_size,
            gfpgan_weight=gfpgan_weight,
        )

        try:
            first_frame = next(frame_generator)
        except StopIteration:
            raise ValueError("No frames generated.")

        height, width, _ = first_frame.shape

        # Direct FFMPEG stdin pipe - bypass writing frame PNGs to disk!
        temp_dir = os.path.join(result_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_vid_path = os.path.join(temp_dir, f"temp_{input_basename}_{audio_basename}.mp4")

        cmd_img2video = [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            temp_vid_path
        ]

        print("Piping generated frames directly to FFMPEG process...")
        process = subprocess.Popen(cmd_img2video, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Write first frame
        process.stdin.write(first_frame.tobytes())

        # Write remaining frames
        for frame in tqdm(frame_generator, desc="Blending & Piping"):
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        # Combine video and audio
        cmd_combine_audio = [
            "ffmpeg", "-y", "-v", "warning",
            "-i", audio_path,
            "-i", temp_vid_path,
            "-c:v", "copy", "-c:a", "aac",
            output_vid_name
        ]
        subprocess.run(cmd_combine_audio, check=True)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        print(f"Results saved to {output_vid_name}")
        return output_vid_name

    def generate_stream(
        self,
        audio_path: str,
        video_path: str,
        enhance: bool = False,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        fps: int = 25,
        batch_size: int = 1,
        gfpgan_weight: float = 0.5,
    ):
        """Generator yielding MJPEG multipart chunks in real-time."""
        frame_generator = self._generate_frames(
            audio_path=audio_path,
            video_path=video_path,
            enhance=enhance,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            fps=fps,
            batch_size=batch_size,
            gfpgan_weight=gfpgan_weight,
        )

        for frame in frame_generator:
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

    def get_gpu_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": None,
            "memory_allocated": None,
            "memory_reserved": None,
            "memory_total": None,
        }
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_reserved"] = torch.cuda.memory_reserved(0)
            props = torch.cuda.get_device_properties(0)
            info["memory_total"] = props.total_memory
        return info
