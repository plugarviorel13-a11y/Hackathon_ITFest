# recorder.py
import cv2
import time
import os
from collections import deque
from config import CADRE_TRECUT, CADRE_VIITOR, FOLDER_ALERTE

class VideoRecorder:
    def __init__(self):
        self.frame_buffer = deque(maxlen=CADRE_TRECUT)
        self.buffer_salvare = []
        self.cadre_inregistrate = 0
        self.in_alerta = False
        
        # Ne asiguram ca folderul exista  daca nu il facem
        if not os.path.exists(FOLDER_ALERTE):
            os.makedirs(FOLDER_ALERTE)

    def adauga_cadru_normal(self, frame):
        self.frame_buffer.append(frame.copy())

    def declanseaza_inregistrarea(self):
        if not self.in_alerta:
            self.in_alerta = True
            self.buffer_salvare = list(self.frame_buffer)
            self.cadre_inregistrate = 0

    def inregistreaza_cadru_alerta(self, frame):
        self.buffer_salvare.append(frame.copy())
        self.cadre_inregistrate += 1

        if self.cadre_inregistrate >= CADRE_VIITOR:
            self._salveaza_fisier()
            self.in_alerta = False 
            self.buffer_salvare = []
            return True 
        return False

    def _salveaza_fisier(self):
        if not self.buffer_salvare: return
        inaltime, latime, _ = self.buffer_salvare[0].shape
        
        # Numele fisierului
        nume_fisier = f"incident_{int(time.time())}.mp4"
        
        # Lipim numele folderului de numele fisierului
        cale_completa = os.path.join(FOLDER_ALERTE, nume_fisier)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cale_completa, fourcc, 30.0, (latime, inaltime))
        
        for cadru in self.buffer_salvare:
            out.write(cadru)
        out.release()
        
        print(f"\n [SUCCES] Raport salvat în folder: {cale_completa}\n")