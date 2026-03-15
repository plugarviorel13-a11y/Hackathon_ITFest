# main.py
import cv2
import time
import threading 
import os
from config import VIDEO_SOURCE
from video_stream import LiveStream
from detector import MotionAnomalyDetector
from recorder import VideoRecorder
from analyzer import LocalOllamaAnalyzer

# Variabile globale
analiza_ai_in_curs = False
status_ai = "Monitorizare Activa"

def procedura_verificare_ai(ai_analyzer, frame, recorder, scor_fizic):
    global analiza_ai_in_curs, status_ai
    
    analiza_ai_in_curs = True
    
    
    if scor_fizic > 15.0:
        print(f" [URGENT] Scorul fizic este {scor_fizic:.1f}%! Trecem peste AI, impact major!")
        status_ai = "!!! IMPACT EXTREM DETECTAT !!!"
        
        if not os.path.exists("ALERTE"): os.makedirs("ALERTE")
        timestamp = time.strftime("%H-%M-%S")
        cv2.imwrite(f"ALERTE/accident_fizica_{timestamp}.jpg", frame)
        recorder.declanseaza_inregistrarea()
        
        time.sleep(4)
        status_ai = "Monitorizare Activa"
        analiza_ai_in_curs = False
        return

   
    status_ai = "AI ANALIZEAZA..."
    raspuns_ai = ai_analyzer.intreaba_ai(frame)
    print(f" Răspuns AI: {raspuns_ai}")
    
    
    cuvinte_accident = ["ACCIDENT", "YES", "CRASH", "COLLISION", "HIT"]
    if any(cuvant in raspuns_ai.upper() for cuvant in cuvinte_accident):
        status_ai = "!!! ACCIDENT CONFIRMAT AI !!!"
        
        if not os.path.exists("ALERTE"): os.makedirs("ALERTE")
        timestamp = time.strftime("%H-%M-%S")
        cv2.imwrite(f"ALERTE/accident_ai_{timestamp}.jpg", frame)
        
        recorder.declanseaza_inregistrarea()
    else:
        status_ai = "Alarma falsa filtrata"
    
    time.sleep(3) 
    status_ai = "Monitorizare Activa"
    analiza_ai_in_curs = False

def start_sistem():
    global analiza_ai_in_curs, status_ai
    
    stream = LiveStream(VIDEO_SOURCE)
    detector = MotionAnomalyDetector()
    recorder = VideoRecorder()
    ai_analyzer = LocalOllamaAnalyzer()
    
    ultimul_check_ai = 0
    cooldown_ai = 15

    cv2.namedWindow('Camera Intersectie', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Intersectie', 800, 450)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    culoare_alerta = (0, 0, 255)
    culoare_ok = (0, 255, 0)
    culoare_ai = (0, 255, 255)

    print("Sistem Hibrid (Fizica + AI cu Fail-Safe) pornit.")

    while True:
        ret, frame = stream.citeste_cadru()
        if not ret: 
            print(" [Sistem] Fisier video terminat. Inchidem aplicatia...")
            break  
        
        miscare_masiva, scor, masca_fizica, cadru_procesat = detector.analizeaza_cadru(frame)
        
        
        timp_curent = time.time()
        if miscare_masiva and not analiza_ai_in_curs and (timp_curent - ultimul_check_ai > cooldown_ai):
            ultimul_check_ai = timp_curent
            print(f"\n Anomalie detectata ({scor:.2f}%). Lansam procedura...")
            
            threading.Thread(target=procedura_verificare_ai, args=(ai_analyzer, frame.copy(), recorder, scor)).start()

        
        cv2.rectangle(cadru_procesat, (0, 0), (450, 40), (0,0,0), -1)
        
        display_color = culoare_ok
        if "ANALIZA" in status_ai: display_color = culoare_ai
        if "IMPACT" in status_ai or "ACCIDENT" in status_ai: display_color = culoare_alerta
        
        cv2.putText(cadru_procesat, f"STATUS: {status_ai}", (10, 25), font, 0.6, display_color, 2)

        if recorder.in_alerta:
            recorder.inregistreaza_cadru_alerta(frame)
            cv2.putText(cadru_procesat, " REC RAPORT", (cadru_procesat.shape[1]-150, 25), font, 0.6, (0,0,255), 2)
        else:
            recorder.adauga_cadru_normal(frame)

        cv2.imshow('Camera Intersectie', cadru_procesat)
        cv2.imshow('Creierul Fizic', masca_fizica)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    stream.opreste()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_sistem()
