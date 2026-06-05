#!/usr/bin/env python3
"""
Test script for MuseTalk API endpoints
"""

import requests
import os
import sys

API_URL = os.getenv("API_URL", "http://localhost:8000")

def test_health():
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_root():
    print("Testing root endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_generate_file_upload(audio_path, video_path):
    print("Testing generate endpoint with file upload...")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    files = {
        "audio": open(audio_path, "rb"),
        "source": open(video_path, "rb")
    }
    data = {
        "bbox_shift": 0,
        "extra_margin": 10,
        "parsing_mode": "jaw",
        "fps": 25,
        "batch_size": 8
    }
    
    print("Sending request...")
    response = requests.post(f"{API_URL}/generate", files=files, data=data)
    
    if response.status_code == 200:
        res_json = response.json()
        print(f"API Response metadata: {res_json}")
        download_url = res_json.get("download_url")
        if download_url:
            full_download_url = f"{API_URL}{download_url}"
            print(f"Downloading video from {full_download_url}...")
            video_resp = requests.get(full_download_url)
            if video_resp.status_code == 200:
                output_file = "test_output_upload.mp4"
                with open(output_file, "wb") as f:
                    f.write(video_resp.content)
                print(f"Success! Video saved to {output_file}")
            else:
                print(f"Failed to download video: {video_resp.status_code}")
        else:
            print("Error: No download_url in response")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
    
    files["audio"].close()
    files["source"].close()
    print()

def test_generate_json(audio_path, video_path):
    print("Testing generate endpoint with JSON...")
    
    # Translate local paths to container paths if they match the mounted volume
    container_audio = audio_path
    container_video = video_path
    
    if audio_path.startswith("data/") or audio_path.startswith("./data/"):
        rel_path = audio_path.replace("./data/", "").replace("data/", "")
        container_audio = f"/app/data/{rel_path}"
        
    if video_path.startswith("data/") or video_path.startswith("./data/"):
        rel_path = video_path.replace("./data/", "").replace("data/", "")
        container_video = f"/app/data/{rel_path}"
        
    print(f"Local paths: {audio_path} -> Container: {container_audio}")
    print(f"Local paths: {video_path} -> Container: {container_video}")
    
    payload = {
        "audio_path": container_audio,
        "video_path": container_video,
        "bbox_shift": 0,
        "extra_margin": 10,
        "parsing_mode": "jaw",
        "fps": 25,
        "batch_size": 8
    }
    
    print("Sending request...")
    response = requests.post(f"{API_URL}/generate/json", json=payload)
    
    if response.status_code == 200:
        res_json = response.json()
        print(f"Success! Response: {res_json}")
        # Download the generated video file if possible
        output_video_path = res_json.get("output_video_path")
        if output_video_path:
            filename = os.path.basename(output_video_path)
            full_download_url = f"{API_URL}/download/{filename}"
            print(f"Downloading video from {full_download_url}...")
            video_resp = requests.get(full_download_url)
            if video_resp.status_code == 200:
                output_file = "test_output_json.mp4"
                with open(output_file, "wb") as f:
                    f.write(video_resp.content)
                print(f"Success! Video saved to {output_file}")
            else:
                print(f"Failed to download video: {video_resp.status_code}")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
    
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("MuseTalk API Test Script")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_root()
        
        if len(sys.argv) >= 3:
            audio_file = sys.argv[1]
            video_file = sys.argv[2]
            test_generate_json(audio_file, video_file)
            test_generate_file_upload(audio_file, video_file)
        else:
            print("To test generation, provide audio and video file paths:")
            print("  python test_api.py <audio_path> <video_path>")
            print()
            print("Example:")
            print("  python test_api.py ./assets/demo/man/man.png ./assets/demo/yongen/yongen.jpeg")
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is the server running?")
        print(f"  API URL: {API_URL}")
    except Exception as e:
        print(f"Error: {e}")
