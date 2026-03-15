# app.py
import cv2
import time
import threading
import logging
import os
import glob
from flask import Flask, render_template, Response, jsonify, send_file

from config import VIDEO_SOURCE
from video_stream import LiveStream
from detector import MotionAnomalyDetector
from analyzer import LocalOllamaAnalyzer
from recorder import VideoRecorder

# Suprimare log-uri standard Flask
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Initializare module principale
stream = LiveStream(VIDEO_SOURCE)
detector = MotionAnomalyDetector()
ai_analyzer = LocalOllamaAnalyzer()
recorder = VideoRecorder()

# Starea curenta a sistemului (partajata intre thread-uri)
current_data = {
    "anomaly": False, "score": 0.0, "type": "NORMAL",
    "masa": 0.0, "delta": 0.0, "latency_ms": 0, 
    "ai_check": False, "ai_description": "" 
}

# Buffer pentru ultimele cadre procesate
latest_frames = {
    "camera": None,
    "brain": None
}

def procedura_ai(frame_crop, scor_fizic):
    """Proceseaza cadrul utilizand modelul local AI si valideaza rezultatul."""
    global current_data
    
    try:
        rezultat_complet = ai_analyzer.intreaba_ai(frame_crop)
        
        # Validare bazata pe cuvinte cheie
        cuvinte_accident = ["CRASH", "COLLISION", "ACCIDENT", "SMASHED", "HIT", "YES", "DAMAGED"]
        ai_zice_da = any(cuvant in rezultat_complet.upper() for cuvant in cuvinte_accident)
        
        if ai_zice_da or scor_fizic > 20.0:
            current_data["type"] = "RAPORT AI COMPLETAT"
            if scor_fizic > 20.0 and not ai_zice_da:
                current_data["ai_description"] = f"[ALARMA FORTATA] AI: {rezultat_complet}"
            else:
                current_data["ai_description"] = rezultat_complet
        else:
            current_data["type"] = "ALARMA FALSA"
            current_data["ai_description"] = rezultat_complet
            
    except Exception as e:
        print(f"[Error] Procedura AI a esuat: {e}")
        current_data["ai_description"] = "Eroare la generarea raportului AI."
        current_data["type"] = "EROARE AI"

    # Mentinere stare de alerta pentru vizualizare interfata (60 secunde)
    time.sleep(60)
    
    current_data["anomaly"] = False
    current_data["type"] = "NORMAL"
    current_data["ai_check"] = False


def procesor_fundal():
    """Bucla principala asincrona pentru procesarea fluxului video."""
    global current_data, latest_frames
    
    ultimul_check_ai = 0
    ultimul_frame_id = -1
    timp_ultimul_cadru = time.time()
    
    print("[Info] Procesor de fundal initializat. Rulare continua activa.")
    
    while True:
        try:
            ret, frame, frame_id = stream.citeste_cadru()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Verificare integritate frame si prevenire procesare duplicata
            if frame_id == ultimul_frame_id:
                time.sleep(0.01)
                continue
            
            ultimul_frame_id = frame_id

            if not recorder.in_alerta:
                recorder.adauga_cadru_normal(frame)

            now = time.time()
            if (now - timp_ultimul_cadru) < 0.04: 
                time.sleep(0.005)
                continue

            timp_ultimul_cadru = now

            # Rulare algoritm detectie miscare
            miscare, scor, masca, cadru_proiectat = detector.analizeaza_cadru(frame)
            
            # Verificare conditii declansare alerta
            if miscare and scor > 20.0 and not current_data["ai_check"] and (now - ultimul_check_ai > 65):
                ultimul_check_ai = now
                current_data["ai_check"] = True
                
                print(f"[Alerta] Impact detectat (Scor: {scor:.1f}%). Fluxul video continua.")
                
                try:
                    recorder.declanseaza_inregistrarea()
                except Exception as e:
                    print(f"[Warning] Eroare initializare inregistrare: {e}")
                
                current_data["anomaly"] = True
                current_data["type"] = "ANOMALIE DETECTATA"
                current_data["ai_description"] = "Asteptam analiza modelului..."
                
                threading.Thread(target=procedura_ai, args=(frame.copy(), scor)).start()

            # Gestionare salvare cadre in MP4
            try:
                if recorder.in_alerta:
                    recorder.inregistreaza_cadru_alerta(frame)
                    cv2.putText(cadru_proiectat, "REC", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                pass 

            # Actualizare date globale
            current_data["score"] = scor / 10
            current_data["masa"] = scor

            cadru_web = cv2.resize(cadru_proiectat, (640, 360))
            masca_web = cv2.resize(masca, (640, 360))
            
            _, buf_cam = cv2.imencode('.jpg', cadru_web, [cv2.IMWRITE_JPEG_QUALITY, 45])
            _, buf_brain = cv2.imencode('.jpg', masca_web, [cv2.IMWRITE_JPEG_QUALITY, 45])
            
            latest_frames["camera"] = buf_cam.tobytes()
            latest_frames["brain"] = buf_brain.tobytes()
            
        except Exception as e:
            print(f"[Error] Exceptie neprevazuta in procesorul de fundal: {e}")
            time.sleep(0.1)

# Pornire procesor analiza video
threading.Thread(target=procesor_fundal, daemon=True).start()

def trimite_flux(feed_type):
    """
    Generator continuu pentru clientul web (Mecanism Heartbeat).
    Transmite ultimul cadru disponibil pentru a mentine conexiunea deschisa.
    """
    while True:
        frame_bytes = latest_frames.get(feed_type)
        if frame_bytes:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Mentinere rata constanta de transmisie
        time.sleep(0.033) 

#Rute Web (Flask)

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video_feed')
def video_feed(): 
    return Response(trimite_flux("camera"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/brain_feed')
def brain_feed(): 
    return Response(trimite_flux("brain"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anomaly_score')
def anomaly_score(): 
    return jsonify(current_data)

@app.route('/download_video')
def download_video():
    folder = "Rapoarte_Politie"
    if not os.path.exists(folder):
        return "Directorul nu exista.", 404
    list_of_files = glob.glob(f'{folder}/*.mp4')
    if not list_of_files:
        return "Fisierul este inca in curs de scriere.", 404
    latest_file = max(list_of_files, key=os.path.getctime)
    return send_file(latest_file, as_attachment=True)

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("[System] Server Flask initializat")
        print("[System] Mod vizualizare continua activ")
        print("[System] Acces: http://127.0.0.1:5000")
        print("="*50 + "\n")
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n[System] Semnal oprire interceptat. Inchidere procese...")
        stream.opreste()
        os._exit(0)