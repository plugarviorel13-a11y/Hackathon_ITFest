# analyzer.py
import cv2
import base64
import requests

class LocalOllamaAnalyzer:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model_name = "moondream"
        print("[Sistem] Moondream Analyzer initializat.")

    def intreaba_ai(self, frame):
        try:
            # Optimizare rezolutie pentru procesare rapida
            resized_frame = cv2.resize(frame, (512, 512))
            cv2.imwrite("static/debug_ai_vision.jpg", resized_frame)

            _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            imagine_b64 = base64.b64encode(buffer).decode('utf-8')
        
            # Prompt 
            prompt = "Briefly describe the vehicles in this image. Is there a car crash or collision happening?"

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [imagine_b64],
                "stream": False,
                "options": {
                    "temperature": 0.4,   # Echilibru intre coerenta gramaticala si focus
                    "num_predict": 80     # SINTAXA CORECTA OLLAMA pentru limitarea textului
                }
            }

            print("[Sistem] AI-ul analizeaza evenimentul...")
           
            response = requests.post(self.url, json=payload, timeout=45)
            
            if response.status_code == 200:
                rezultat_raw = response.json().get("response", "").strip()
                if not rezultat_raw:
                    return "EROARE: AI-ul nu a generat niciun text."
                print(f"[AI RAW]: {rezultat_raw}")
                return rezultat_raw
            else:
                return f"EROARE API Ollama: Cod {response.status_code}"
        except requests.exceptions.Timeout:
            return "EROARE TIMEOUT: Timp de procesare depasit."
        except Exception as e:
            return f"EROARE CONEXIUNE: {str(e)}"