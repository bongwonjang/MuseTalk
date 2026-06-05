#!/usr/bin/env python3
"""
Test script for MuseTalk API streaming endpoints.
It connects to the stream and saves incoming frames to a local directory to verify that they are being generated.
"""

import requests
import os
import sys
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

def test_stream_json(audio_path, video_path, output_dir="./results/streamed_frames", max_frames=15):
    print("=" * 60)
    print("Testing Streaming Endpoint via JSON (multipart/x-mixed-replace)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Frames will be saved to: {os.path.abspath(output_dir)}\n")
    
    # Translate local paths to container paths if they match mounted volumes
    container_audio = audio_path
    container_video = video_path
    if audio_path.startswith("data/") or audio_path.startswith("./data/"):
        rel_path = audio_path.replace("./data/", "").replace("data/", "")
        container_audio = f"/app/data/{rel_path}"
    if video_path.startswith("data/") or video_path.startswith("./data/"):
        rel_path = video_path.replace("./data/", "").replace("data/", "")
        container_video = f"/app/data/{rel_path}"
        
    payload = {
        "audio_path": container_audio,
        "video_path": container_video,
        "bbox_shift": 0,
        "extra_margin": 10,
        "parsing_mode": "jaw",
        "fps": 25,
        "batch_size": 8
    }
    
    url = f"{API_URL}/generate/stream/json"
    print(f"POST {url}")
    print(f"Payload: {payload}\n")
    
    start_time = time.time()
    
    try:
        # Use stream=True to read the response iteratively
        response = requests.post(url, json=payload, stream=True)
        if response.status_code != 200:
            print(f"Error: Server returned status {response.status_code}")
            print(response.text)
            return
            
        print("Connected to stream. Parsing MJPEG frames...")
        
        # Buffer to accumulate incoming byte chunks
        byte_data = b""
        frame_count = 0
        
        # Iterate over the response stream
        for chunk in response.iter_content(chunk_size=4096):
            byte_data += chunk
            
            # Find the boundaries in the multipart response
            while True:
                # Find start of frame
                a = byte_data.find(b"\xff\xd8")  # JPEG start-of-image (SOI) marker
                b = byte_data.find(b"\xff\xd9")  # JPEG end-of-image (EOI) marker
                
                if a != -1 and b != -1 and a < b:
                    # Extract the full JPEG bytes
                    jpg_bytes = byte_data[a : b + 2]
                    
                    # Remove the processed part from the buffer
                    byte_data = byte_data[b + 2 :]
                    
                    # Save frame
                    frame_path = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
                    with open(frame_path, "wb") as f:
                        f.write(jpg_bytes)
                    
                    elapsed = time.time() - start_time
                    print(f"  [Frame {frame_count:03d}] Received & saved. Size: {len(jpg_bytes)} bytes. Time from start: {elapsed:.2f}s")
                    frame_count += 1
                    
                    if frame_count >= max_frames:
                        print(f"\nReached max_frames limit ({max_frames}). Closing stream connection.")
                        response.close()
                        return
                else:
                    # We don't have a complete frame in the buffer yet, read more chunks
                    break
                    
    except KeyboardInterrupt:
        print("\nStream test interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    audio = "data/audio/yongen.wav"
    video = "data/video/yongen.mp4"
    if len(sys.argv) >= 3:
        audio = sys.argv[1]
        video = sys.argv[2]
        
    test_stream_json(audio, video)
