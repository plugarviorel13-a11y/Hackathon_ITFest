# VisionSentinel 

**Autonomous Edge AI Traffic Anomaly Detection**

VisionSentinel transforms standard, passive city surveillance cameras (RTSP streams) into intelligent, autonomous sensors. Built for edge computing, it detects car crashes, severe traffic anomalies, and hazards in real-time (under 20 seconds) without relying on human dispatchers or continuous cloud streaming.

##  The Problem & The Solution
Cities monitor traffic using thousands of "dumb" cameras, relying on human operators to spot crashes or waiting for 112 emergency calls. Every lost minute costs lives. 

**VisionSentinel** solves this by acting as a 24/7 digital dispatcher. It actively watches the feed, detects physical anomalies using lightweight computer vision, validates the event using a local Vision Language Model (VLM), and instantly alerts authorities with a GPS pin, a video clip, and an AI-generated text report.

## How It Works (Hybrid Pipeline)
To ensure the system runs smoothly on edge devices without overheating the CPU, we built a two-stage hybrid pipeline:

1. **Filter 1: The Fast Worker (OpenCV & MOG2)**
   - Runs constantly at 30 FPS using <1% CPU.
   - Uses Background Subtraction to track the "mass" and "geometry" of moving pixels.
   - When cars collide, the motion geometry changes violently, generating a mathematical anomaly (Delta Score).
   - If the score crosses the danger threshold, it triggers the AI.

2. **Filter 2: The Validating Brain (Moondream VLM via Ollama)**
   - When an anomaly is detected, a 512x512 compressed frame is sent to our local AI model.
   - The AI acts as an anti-false-alarm shield, answering a strict prompt: *"Is there a car crash or collision in this image?"*
   - If confirmed, the system triggers the web dashboard alert, drops a pin on the live map, and auto-saves the video evidence (MP4).

## Tech Stack
* **Backend:** Python, Flask, OpenCV (cv2)
* **AI / Deep Learning:** Ollama (Local LLM Server), Moondream (Vision Language Model)
* **Frontend:** HTML5, Custom CSS Variables (Dark/Cyberpunk UI), Vanilla JavaScript (Async Polling)
* **Mapping & Analytics:** Leaflet.js (Live GPS mapping), Chart.js (Real-time anomaly scoring)

## 🛠️ Installation & Setup

**1. Install Local AI (Ollama)**
You must have Ollama installed on your system to run the local vision model.
```bash
# Pull the Moondream model
ollama run moondream
