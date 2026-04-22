from ultralytics import YOLO
import cv2
import time
from collections import deque

class TERPassengerCounter:
    def __init__(self):
        """Initialise le détecteur avec des paramètres optimisés pour les environnements TER"""
        # Chargement du modèle (téléchargement automatique si absent)
        self.model = YOLO('yolov8m.pt')  # Version medium pour meilleure précision
        
        # Configuration de la détection
        self.class_ids = [0]  # 0 = personne dans COCO
        self.conf_threshold = 0.6  # Seuil de confiance
        self.iou_threshold = 0.45  # Seuil de recouvrement
        
        # Statistiques
        self.frame_count = 0
        self.total_passengers = 0
        self.fps_history = deque(maxlen=30)
        
        # Style d'affichage
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.box_thickness = 2
        self.text_color = (0, 255, 0)  # Vert
        self.box_color = (0, 0, 255)  # Rouge
        
    def process_frame(self, frame):
        """Traite une frame vidéo et retourne le résultat annoté"""
        start_time = time.time()
        
        # Détection avec YOLOv8
        results = self.model.predict(
            frame,
            classes=self.class_ids,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Calcul du FPS
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Récupération des résultats
        annotated_frame = results[0].plot(
            line_width=self.box_thickness,
            font_size=self.font_scale,
            pil=False
        )
        
        current_passengers = len(results[0].boxes)
        self.total_passengers += current_passengers
        self.frame_count += 1
        
        # Affichage des informations
        cv2.putText(annotated_frame, f"Passagers: {current_passengers}", 
                   (10, 30), self.font, self.font_scale, self.text_color, self.font_thickness)
        cv2.putText(annotated_frame, f"Total: {self.total_passengers}", 
                   (10, 60), self.font, self.font_scale, self.text_color, self.font_thickness)
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                   (10, 90), self.font, self.font_scale, self.text_color, self.font_thickness)
        
        return annotated_frame

def main():
    # Initialisation
    detector = TERPassengerCounter()
    
    # Sources vidéo possibles
    video_source = "ter_video.mp4"  # Chemin vers fichier vidéo
    # video_source = 0  # Pour utiliser la webcam
    
    # Initialisation du flux vidéo
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la source vidéo {video_source}")
        return
    
    # Configuration de l'enregistrement (optionnel)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_file = "output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    # Boucle principale
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Traitement de la frame
        processed_frame = detector.process_frame(frame)
        
        # Enregistrement (optionnel)
        out.write(processed_frame)
        
        # Affichage
        cv2.imshow('Détection Passagers TER - YOLOv8', processed_frame)
        
        # Arrêt avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Nettoyage
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Traitement terminé. Total passagers détectés: {detector.total_passengers}")

if __name__ == "__main__":
    main()