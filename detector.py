# detector.py
import cv2
import numpy as np
from config import REZOLUTIE_PROCESARE, THRESHOLD_ANOMALIE

class MotionAnomalyDetector:
    def __init__(self):
    
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    def analizeaza_cadru(self, frame):
        cadru_procesat = frame.copy()
        frame_mic = cv2.resize(frame, REZOLUTIE_PROCESARE)
        
        # extragem miscarea 
        fgmask = self.fgbg.apply(frame_mic)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        h, w = fgmask.shape
        margine_y = int(h * 0.15) 
        margine_x = int(w * 0.25) 

        # stergem absolut orice miscare de pe margini 
        fgmask[0:margine_y, :] = 0
        fgmask[h-margine_y:h, :] = 0
        fgmask[:, 0:margine_x] = 0
        fgmask[:, w-margine_x:w] = 0

        # Desenam cutia de ROI 
        scale_x = frame.shape[1] / w
        scale_y = frame.shape[0] / h
        p1 = (int(margine_x * scale_x), int(margine_y * scale_y))
        p2 = (int((w - margine_x) * scale_x), int((h - margine_y) * scale_y))
        cv2.rectangle(cadru_procesat, p1, p2, (255, 255, 0), 1)
        cv2.putText(cadru_procesat, "ROI - ZONA DE IMPACT", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        
        contururi, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        aria_totala_miscare = 0
        
        for contur in contururi:
            aria = cv2.contourArea(contur)
            if aria > 400: 
                aria_totala_miscare += aria
                
                # Coordonatele pentru UI
                x, y, w_box, h_box = cv2.boundingRect(contur)
                x_orig = int(x * scale_x)
                y_orig = int(y * scale_y)
                w_orig = int(w_box * scale_x)
                h_orig = int(h_box * scale_y)
                
                # Desenam UI-ul  
                cv2.rectangle(cadru_procesat, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (255, 255, 0), 1)
                
                centru_x = x_orig + (w_orig // 2)
                centru_y = y_orig + (h_orig // 2)
                cv2.circle(cadru_procesat, (centru_x, centru_y), 4, (0, 0, 255), -1)
                cv2.line(cadru_procesat, (x_orig, y_orig), (centru_x, centru_y), (255, 255, 0), 1)
                cv2.putText(cadru_procesat, "TARGET", (x_orig, y_orig - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 3. Calculam procentul real de miscare raportat strict la centrul ecranului
        total_pixeli_centru = (w - 2 * margine_x) * (h - 2 * margine_y)
        procent_miscare = (aria_totala_miscare / total_pixeli_centru) * 100
        
        este_anomalie = procent_miscare > THRESHOLD_ANOMALIE
        
        
        cv2.putText(cadru_procesat, f"Masa in ROI: {procent_miscare:.2f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return este_anomalie, procent_miscare, fgmask, cadru_procesat