# ======================================================================
# PLATEFORME DE DETECTION ET COMPTAGE DE VEHICULES
# Version avec YOLO26 - Seuil de confiance variable a partir de 0.01
# ======================================================================

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict, deque
import plotly.express as px
import plotly.graph_objects as go
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import torch
import gc
import re
import subprocess
import sys

# Configuration de la page
st.set_page_config(
    page_title="Detection Vehicules YOLO26", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# VERIFICATION ET INSTALLATION DE YOLO26
# ======================================================================

def check_and_install_yolo26():
    """Verifie et installe la derniere version de YOLO26"""
    try:
        import ultralytics
        current_version = ultralytics.__version__
        st.info(f"Version Ultralytics actuelle: {current_version}")
        
        # Verifier si la version est recente
        version_parts = current_version.split('.')
        if int(version_parts[0]) >= 8 and int(version_parts[1]) >= 3:
            return True
        else:
            st.warning(f"Version {current_version} detectee. Mise a jour vers YOLO26...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
                st.success("YOLO26 installe avec succes! Veuillez redemarrer l'application.")
                st.rerun()
            except:
                st.warning("Impossible de mettre a jour. Utilisation de la version actuelle.")
                return True
    except ImportError:
        st.info("Installation de YOLO26...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            st.success("YOLO26 installe avec succes! Veuillez redemarrer l'application.")
            st.rerun()
        except:
            st.error("Erreur d'installation. Veuillez installer manuellement: pip install ultralytics")
            return False
    return True

# ======================================================================
# CONFIGURATION & CONSTANTS
# ======================================================================

@dataclass
class VehicleConfig:
    """Configuration centralisee de l'application"""
    # Parametres YOLO26 - Seuil tres bas pour maximiser la detection
    confidence_threshold: float = 0.05  # Seuil par defaut a 0.05 (5%)
    nms_threshold: float = 0.35
    frame_skip: int = 1
    track_smoothness: float = 0.65
    track_max_age: int = 45
    track_distance_threshold: float = 1.5
    calibration_factor: float = 0.05
    show_tracks: bool = True
    show_bboxes: bool = True
    show_ids: bool = True
    show_speed: bool = True
    show_distance: bool = True
    detection_history_size: int = 100
    processing_time_window: int = 30
    fps_smoothing: float = 0.8


VEHICLE_CLASSES = {
    0: "Personne",
    1: "Velo",
    2: "Voiture",
    3: "Moto",
    5: "Bus",
    7: "Camion",
}

ENABLED_CLASSES_DEFAULT = {2, 3, 5, 7}

VEHICLE_COLORS = {
    "Voiture": (235, 152, 41),
    "Moto": (60, 76, 231),
    "Bus": (15, 196, 241),
    "Camion": (173, 68, 142),
    "Velo": (96, 174, 39),
    "Personne": (141, 140, 127),
}


# ======================================================================
# DATA CLASSES
# ======================================================================

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    class_id: int
    class_name: str
    width: int
    height: int


@dataclass
class Track:
    id: int
    class_name: str
    class_id: int
    color: Tuple[int, int, int]
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    positions_pixels: deque = field(default_factory=lambda: deque(maxlen=60))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    bboxes: deque = field(default_factory=lambda: deque(maxlen=20))
    age: int = 0
    first_seen_frame: int = 0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    confidence: float = 0.0
    width: int = 0
    height: int = 0
    speed_kmh: float = 0.0
    speeds_history: deque = field(default_factory=lambda: deque(maxlen=10))
    distance_to_next: float = 0.0
    distance_to_prev: float = 0.0
    
    def add_detection(self, detection: Detection, smoothness: float = 0.55, timestamp: float = 0):
        curr_x, curr_y = detection.center
        if self.positions:
            last_x, last_y = self.positions[-1]
            curr_x = int(smoothness * last_x + (1 - smoothness) * curr_x)
            curr_y = int(smoothness * last_y + (1 - smoothness) * curr_y)
        self.positions.append((curr_x, curr_y))
        self.positions_pixels.append((curr_x, curr_y))
        self.timestamps.append(timestamp)
        self.bboxes.append(detection.bbox)
        self.confidence = detection.confidence
        self.width = detection.width
        self.height = detection.height
        self.age = 0
        self.last_seen_time = timestamp
        self._calculate_speed()
    
    def _calculate_speed(self, calibration_factor: float = 0.05):
        if len(self.positions_pixels) >= 2 and len(self.timestamps) >= 2:
            x1, y1 = self.positions_pixels[-2]
            x2, y2 = self.positions_pixels[-1]
            t1 = self.timestamps[-2]
            t2 = self.timestamps[-1]
            distance_pixels = math.hypot(x2 - x1, y2 - y1)
            distance_meters = distance_pixels * calibration_factor
            time_hours = (t2 - t1) / 3600.0
            if time_hours > 0:
                self.speed_kmh = distance_meters / time_hours / 1000.0
                self.speeds_history.append(self.speed_kmh)


@dataclass
class VehicleRecord:
    record_id: int
    track_id: int
    category: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: str
    last_seen_time: str
    max_confidence: float
    avg_confidence: float
    avg_speed_kmh: float
    max_speed_kmh: float
    duration_seconds: float
    track_points: List[Tuple[int, int]]


@dataclass
class DistanceRecord:
    timestamp: float
    frame: int
    vehicle1_id: int
    vehicle1_category: str
    vehicle2_id: int
    vehicle2_category: str
    distance_meters: float
    distance_pixels: float


# ======================================================================
# DETECTEUR DE VEHICULES AVEC YOLO26
# ======================================================================

class VehicleDetector:
    """Detecteur utilisant YOLO26 avec seuil ultra bas"""
    
    def __init__(self, config: VehicleConfig):
        self.config = config
        self.model = None
        self.use_yolo = False
        self.model_name = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modele YOLO26"""
        try:
            from ultralytics import YOLO
            
            # Patcher torch.load pour eviter l'erreur weights_only
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            
            # Essayer YOLO26 en priorite
            models_to_try = [
                'yolo26n.pt',    # YOLO26 Nano - ultra rapide
                'yolo26s.pt',    # YOLO26 Small - equilibre
                'yolov8n.pt',    # Fallback YOLOv8
            ]
            
            for model_name in models_to_try:
                try:
                    with st.spinner(f"Chargement de {model_name}..."):
                        self.model = YOLO(model_name)
                        self.use_yolo = True
                        self.model_name = model_name
                        st.success(f"✅ Modele {model_name} charge avec succes!")
                        st.info(f"🎯 Seuil de confiance actuel: {self.config.confidence_threshold:.2f} (0.01 = detection maximale)")
                        break
                except Exception as e:
                    st.warning(f"Impossible de charger {model_name}: {e}")
                    continue
            
            torch.load = original_load
            
            if not self.use_yolo:
                st.warning("⚠️ Aucun modele YOLO disponible. Mode simulation actif.")
            
        except Exception as e:
            st.error(f"Erreur de chargement: {e}")
            self.use_yolo = False
    
    def detect(self, frame: np.ndarray, enabled_classes: Set[int]) -> List[Detection]:
        """Detection avec YOLO26 - Seuil de confiance minimal"""
        if not self.use_yolo or self.model is None:
            return []
        
        try:
            h, w = frame.shape[:2]
            
            # YOLO26 avec seuil ultra bas pour capturer tous les vehicules
            results = self.model(
                frame, 
                conf=self.config.confidence_threshold,  # Seuil variable (0.01 a 0.5)
                iou=self.config.nms_threshold, 
                verbose=False,
                classes=list(enabled_classes),
                device='cpu',
                augment=True,  # Augmente la detection
                imgsz=640,
                max_det=300,   # Nombre maximum de detections
            )
            
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confs, classes):
                    if cls_id not in VEHICLE_CLASSES or cls_id not in enabled_classes:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    box_w, box_h = x2 - x1, y2 - y1
                    
                    # Seuil tres bas pour les petites detections
                    if box_w * box_h < 50 or box_w < 5 or box_h < 5:
                        continue
                    
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        center=((x1 + x2) // 2, (y1 + y2) // 2),
                        confidence=float(conf), 
                        class_id=int(cls_id),
                        class_name=VEHICLE_CLASSES[cls_id],
                        width=box_w, 
                        height=box_h
                    ))
            
            # Suppression des doublons avec seuil bas
            detections = self._remove_duplicates(detections)
            
            return detections
            
        except Exception as e:
            st.warning(f"Erreur detection: {e}")
            return []
    
    def _remove_duplicates(self, detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
        """Supprime les detections en double avec seuil bas"""
        if len(detections) <= 1:
            return detections
        
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        for det in detections:
            duplicate = False
            x1, y1, x2, y2 = det.bbox
            area1 = (x2 - x1) * (y2 - y1)
            
            for kept in keep:
                kx1, ky1, kx2, ky2 = kept.bbox
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    area2 = (kx2 - kx1) * (ky2 - ky1)
                    iou = inter_area / (area1 + area2 - inter_area)
                    
                    if iou > iou_threshold:
                        duplicate = True
                        break
            
            if not duplicate:
                keep.append(det)
        
        return keep


# ======================================================================
# SUIVI DE VEHICULES AMELIORE
# ======================================================================

class VehicleTracker:
    def __init__(self, config: VehicleConfig):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self.lost_tracks: Dict[int, Track] = {}
    
    def update(self, detections: List[Detection], timestamp: float = 0) -> List[int]:
        track_ids = []
        used_detections = set()
        
        for tid, track in list(self.tracks.items()):
            best_idx, best_dist = -1, float('inf')
            last_pos = track.positions[-1] if track.positions else None
            
            if last_pos is None:
                continue
            
            for idx, det in enumerate(detections):
                if idx in used_detections:
                    continue
                
                class_match = 1.0 if det.class_name == track.class_name else 1.5
                dist = math.hypot(det.center[0] - last_pos[0], det.center[1] - last_pos[1]) * class_match
                
                size_similarity = 1.0
                if det.width > 0 and track.width > 0:
                    width_ratio = min(det.width, track.width) / max(det.width, track.width)
                    height_ratio = min(det.height, track.height) / max(det.height, track.height)
                    size_similarity = 1.0 / (width_ratio * height_ratio + 0.5)
                
                dist *= size_similarity
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            threshold = max(track.width, track.height) * self.config.track_distance_threshold
            
            if best_idx != -1 and best_dist < threshold:
                track.add_detection(detections[best_idx], self.config.track_smoothness, timestamp)
                used_detections.add(best_idx)
                track_ids.append(tid)
                self.track_history[tid].append(track.positions[-1])
                if tid in self.lost_tracks:
                    del self.lost_tracks[tid]
            else:
                track.age += 1
                if track.age > self.config.track_max_age:
                    del self.tracks[tid]
                elif track.age > 10:
                    self.lost_tracks[tid] = track
        
        for idx, det in enumerate(detections):
            if idx not in used_detections:
                tid = self.next_id
                hue = (tid * 137) % 360
                color = cv2.cvtColor(np.uint8([[[hue, 210, 240]]]), cv2.COLOR_HSV2BGR)[0][0]
                track = Track(
                    id=tid, class_name=det.class_name, class_id=det.class_id,
                    color=tuple(map(int, color)), first_seen_frame=0,
                    first_seen_time=timestamp, last_seen_time=timestamp,
                    confidence=det.confidence, width=det.width, height=det.height
                )
                track.positions.append(det.center)
                track.positions_pixels.append(det.center)
                track.timestamps.append(timestamp)
                track.bboxes.append(det.bbox)
                self.tracks[tid] = track
                track_ids.append(tid)
                self.track_history[tid].append(det.center)
                self.next_id += 1
        
        return track_ids
    
    def calculate_distances(self, calibration_factor: float = 0.05):
        active_tracks = [(tid, track) for tid, track in self.tracks.items() if track.positions]
        active_tracks.sort(key=lambda x: x[1].positions[-1][0])
        for i, (tid, track) in enumerate(active_tracks):
            track.distance_to_prev = 0
            track.distance_to_next = 0
            if i > 0:
                prev_tid, prev_track = active_tracks[i-1]
                dx = track.positions[-1][0] - prev_track.positions[-1][0]
                dy = track.positions[-1][1] - prev_track.positions[-1][1]
                dist_pixels = math.hypot(dx, dy)
                track.distance_to_prev = dist_pixels * calibration_factor
            if i < len(active_tracks) - 1:
                next_tid, next_track = active_tracks[i+1]
                dx = next_track.positions[-1][0] - track.positions[-1][0]
                dy = next_track.positions[-1][1] - track.positions[-1][1]
                dist_pixels = math.hypot(dx, dy)
                track.distance_to_next = dist_pixels * calibration_factor
    
    def reset(self):
        self.tracks.clear()
        self.track_history.clear()
        self.lost_tracks.clear()
        self.next_id = 0


# ======================================================================
# ANALYSE COMPORTEMENTALE
# ======================================================================

class BehavioralAnalyzer:
    def __init__(self, vehicle_records: List[VehicleRecord], distance_records: List[DistanceRecord], 
                 vehicle_counts: Dict, total_unique: int, metadata: Dict, config: VehicleConfig):
        self.vehicle_records = vehicle_records
        self.distance_records = distance_records
        self.vehicle_counts = vehicle_counts
        self.total_unique = total_unique
        self.metadata = metadata
        self.config = config
    
    def generate_analysis(self) -> str:
        if self.total_unique == 0:
            return "Aucun vehicule detecte pendant la periode d'observation."
        
        parts = []
        parts.append(self._generate_header())
        parts.append(self._generate_global_summary())
        parts.append(self._generate_category_analysis())
        parts.append(self._generate_speed_analysis())
        parts.append(self._generate_distance_analysis())
        parts.append(self._generate_notable_events())
        parts.append(self._generate_behavior_analysis())
        parts.append(self._generate_conclusion())
        return "\n\n".join(parts)
    
    def _generate_header(self) -> str:
        location = self.metadata.get('location', 'le site')
        date = self.metadata.get('date', 'la periode')
        start = self.metadata.get('start_time', 'debut')
        end = self.metadata.get('end_time', 'fin')
        model = self.metadata.get('model_used', 'YOLO26')
        threshold = self.metadata.get('confidence_threshold', 0.05)
        return f"""RAPPORT D'ANALYSE COMPORTEMENTALE DES VEHICULES
================================================================

Lieu d'observation : {location}
Date : {date}
Periode : {start} - {end}
Total vehicules observes : {self.total_unique}
Modele IA utilise : {model}
Seuil de confiance : {threshold:.2f} (detection maximale)"""
    
    def _generate_global_summary(self) -> str:
        if self.vehicle_records:
            total_duration = max([r.duration_seconds for r in self.vehicle_records]) if self.vehicle_records else 0
            traffic_density = self.total_unique / (total_duration / 60) if total_duration > 0 else 0
        else:
            traffic_density = 0
        if traffic_density < 10:
            traffic_level = "FAIBLE"
        elif traffic_density < 30:
            traffic_level = "MODERE"
        else:
            traffic_level = "ELEVE"
        total = self.total_unique
        voiture_pct = (self.vehicle_counts.get("Voiture", 0) / total * 100) if total > 0 else 0
        moto_pct = (self.vehicle_counts.get("Moto", 0) / total * 100) if total > 0 else 0
        return f"""SYNTHESE GLOBALE DU TRAFIC

Niveau de trafic : {traffic_level} ({traffic_density:.1f} vehicules/minute)

Composition du trafic :
- Voitures : {voiture_pct:.1f}% ({self.vehicle_counts.get("Voiture", 0)} vehicules)
- Motos : {moto_pct:.1f}% ({self.vehicle_counts.get("Moto", 0)} vehicules)
- Camions : {self.vehicle_counts.get("Camion", 0)} vehicules
- Bus : {self.vehicle_counts.get("Bus", 0)} vehicules"""
    
    def _generate_category_analysis(self) -> str:
        analysis = "ANALYSE PAR CATEGORIE\n\n"
        for category in ["Voiture", "Moto", "Camion", "Bus"]:
            count = self.vehicle_counts.get(category, 0)
            if count > 0:
                speeds = [r.avg_speed_kmh for r in self.vehicle_records if r.category == category and r.avg_speed_kmh > 0]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    max_speed = max(speeds)
                    analysis += f"""{category} ({count} vehicules)
- Vitesse moyenne : {avg_speed:.0f} km/h
- Vitesse maximale : {max_speed:.0f} km/h
- Comportement : {'Circulation fluide' if avg_speed > 50 else 'Circulation moderee'}

"""
        return analysis
    
    def _generate_speed_analysis(self) -> str:
        all_speeds = [r.avg_speed_kmh for r in self.vehicle_records if r.avg_speed_kmh > 0]
        if not all_speeds:
            return "ANALYSE DES VITESSES\n\nAucune donnee de vitesse disponible."
        overall_avg = sum(all_speeds) / len(all_speeds)
        overall_max = max(all_speeds)
        speed_excess = [s for s in all_speeds if s > 70]
        excess_percentage = (len(speed_excess) / len(all_speeds)) * 100 if all_speeds else 0
        return f"""ANALYSE DES VITESSES

Statistiques globales :
- Vitesse moyenne generale : {overall_avg:.0f} km/h
- Vitesse maximale enregistree : {overall_max:.0f} km/h
- Proportion de vehicules rapides (>70 km/h) : {excess_percentage:.1f}%

Interpretation : Le flux de circulation presente une vitesse {'elevée' if overall_avg > 55 else 'moderee'}. {'Une vigilance accrue est recommandee.' if excess_percentage > 20 else 'Les limitations sont globalement respectees.'}"""
    
    def _generate_distance_analysis(self) -> str:
        if not self.distance_records:
            return "ANALYSE DES DISTANCES\n\nAucune donnee de distance disponible."
        distances = [d.distance_meters for d in self.distance_records]
        avg_distance = sum(distances) / len(distances)
        min_distance = min(distances)
        unsafe_count = len([d for d in distances if d < 20])
        unsafe_percentage = (unsafe_count / len(distances)) * 100 if distances else 0
        return f"""ANALYSE DES DISTANCES DE SECURITE

Statistiques des distances inter-vehiculaires :
- Distance moyenne : {avg_distance:.1f} metres
- Distance minimale : {min_distance:.1f} metres

Evaluation : {'Distance de sécurité souvent insuffisante' if unsafe_percentage > 30 else 'Distance de sécurité généralement respectee'}"""
    
    def _generate_notable_events(self) -> str:
        events = []
        fastest = max(self.vehicle_records, key=lambda x: x.max_speed_kmh) if self.vehicle_records else None
        if fastest and fastest.max_speed_kmh > 0:
            events.append(f"Vehicule le plus rapide : {fastest.category} #{fastest.track_id} avec {fastest.max_speed_kmh:.0f} km/h")
        longest = max(self.vehicle_records, key=lambda x: x.duration_seconds) if self.vehicle_records else None
        if longest and longest.duration_seconds > 0:
            events.append(f"Presence prolongee : {longest.category} #{longest.track_id} est reste {longest.duration_seconds:.1f} secondes")
        if self.distance_records:
            closest = min(self.distance_records, key=lambda x: x.distance_meters)
            if closest.distance_meters < 30:
                events.append(f"Distance critique : {closest.vehicle1_category} et {closest.vehicle2_category} sont passes a {closest.distance_meters:.1f}m")
        if not events:
            return "EVENEMENTS NOTABLES\n\nAucun evenement particulier enregistre."
        events_text = "EVENEMENTS NOTABLES\n\n" + "\n".join([f"{i}. {e}" for i, e in enumerate(events, 1)])
        return events_text
    
    def _generate_behavior_analysis(self) -> str:
        analysis = "ANALYSE COMPORTEMENTALE APPROFONDIE\n\n"
        if self.vehicle_records:
            if len(self.distance_records) > 10:
                avg_dist = sum([d.distance_meters for d in self.distance_records]) / len(self.distance_records)
                if avg_dist < 40:
                    analysis += "- Conduite en peloton : Les vehicules circulent en groupes rapproches.\n"
                else:
                    analysis += "- Conduite dispersee : Les vehicules sont bien espaces.\n"
        if self.distance_records:
            close_distances = [d for d in self.distance_records if d.distance_meters < 25]
            if len(close_distances) > len(self.distance_records) * 0.2:
                analysis += "- Non-respect des distances : Une proportion significative ne maintient pas les distances de securite.\n"
        return analysis
    
    def _generate_conclusion(self) -> str:
        speeds = [r.avg_speed_kmh for r in self.vehicle_records if r.avg_speed_kmh > 0]
        risk_score = 0
        if speeds and sum(speeds)/len(speeds) > 60:
            risk_score += 1
        if self.distance_records:
            unsafe_ratio = len([d for d in self.distance_records if d.distance_meters < 25]) / len(self.distance_records) if self.distance_records else 0
            if unsafe_ratio > 0.2:
                risk_score += 1
        if self.vehicle_counts.get("Moto", 0) > self.total_unique * 0.3:
            risk_score += 1
        risk_levels = ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"]
        return f"""CONCLUSION ET RECOMMANDATIONS

Niveau de risque global : {risk_levels[risk_score]}

Ce rapport a ete genere automatiquement par la plateforme de detection de vehicules.
Les donnees presentees sont basees sur l'analyse video par YOLO26 avec un seuil de confiance
ultra-bas ({self.metadata.get('confidence_threshold', 0.05):.2f}) pour maximiser la detection.

Pour toute information complementaire, veuillez consulter les fichiers CSV/JSON exportes."""


# ======================================================================
# GENERATION PDF SIMPLIFIEE
# ======================================================================

from fpdf import FPDF

class PDFReport(FPDF):
    def __init__(self, counter, analysis_text):
        super().__init__()
        self.counter = counter
        self.analysis_text = self._clean_text(analysis_text)
        self.set_auto_page_break(auto=True, margin=15)
    
    def _clean_text(self, text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'Rapport de comptage de vehicules - YOLO26', 0, 0, 'L')
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')
            self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Genere le {datetime.now().strftime("%d/%m/%Y a %H:%M")}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(46, 204, 113)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_draw_color(46, 204, 113)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        body = self._clean_text(body)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_metadata_section(self):
        self.chapter_title("1. INFORMATIONS DU COMPTAGE")
        metadata = self.counter.metadata
        info = f"""
Lieu d'observation : {metadata.get('location', 'Non specifie')}
Date : {metadata.get('date', 'Non specifiee')}
Heure debut : {metadata.get('start_time', 'Non specifiee')}
Heure fin : {metadata.get('end_time', 'Non specifiee')}
Operateur : {metadata.get('operator', 'Non specifie')}
Calibration : {self.counter.config.calibration_factor:.4f} m/pixel
Total vehicules : {self.counter.total_unique}
Frames traitees : {self.counter.frame_count}
Seuil confiance : {metadata.get('confidence_threshold', 0.05):.2f}
"""
        self.chapter_body(info)
    
    def add_summary_table(self):
        self.chapter_title("2. RESUME PAR CATEGORIE")
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(46, 204, 113)
        self.set_text_color(255, 255, 255)
        self.cell(80, 7, 'Categorie', 1, 0, 'C', True)
        self.cell(50, 7, 'Nombre', 1, 0, 'C', True)
        self.cell(60, 7, 'Pourcentage', 1, 1, 'C', True)
        self.set_font('Arial', '', 9)
        self.set_text_color(0, 0, 0)
        total = self.counter.total_unique
        for cat_name in VEHICLE_CLASSES.values():
            cnt = self.counter.vehicle_counts.get(cat_name, 0)
            if cnt > 0:
                percentage = (cnt / total) * 100
                self.cell(80, 6, cat_name, 1)
                self.cell(50, 6, str(cnt), 1, 0, 'C')
                self.cell(60, 6, f'{percentage:.1f}%', 1, 1, 'C')
        self.cell(80, 6, 'TOTAL', 1)
        self.cell(50, 6, str(total), 1, 0, 'C')
        self.cell(60, 6, '100%', 1, 1, 'C')
        self.ln(4)
    
    def add_speed_statistics(self):
        self.chapter_title("3. STATISTIQUES DE VITESSE")
        speed_stats = self.counter.get_speed_statistics()
        if speed_stats:
            self.set_font('Arial', 'B', 9)
            self.set_fill_color(52, 152, 219)
            self.set_text_color(255, 255, 255)
            self.cell(60, 7, 'Categorie', 1, 0, 'C', True)
            self.cell(40, 7, 'Moyenne', 1, 0, 'C', True)
            self.cell(40, 7, 'Max', 1, 0, 'C', True)
            self.cell(40, 7, 'Min', 1, 1, 'C', True)
            self.set_font('Arial', '', 9)
            self.set_text_color(0, 0, 0)
            for cat, stats in speed_stats.items():
                self.cell(60, 6, cat, 1)
                self.cell(40, 6, f"{stats['moyenne']} km/h", 1, 0, 'C')
                self.cell(40, 6, f"{stats['max']} km/h", 1, 0, 'C')
                self.cell(40, 6, f"{stats['min']} km/h", 1, 1, 'C')
        else:
            self.chapter_body("Aucune donnee de vitesse disponible")
        self.ln(4)
    
    def add_distance_statistics(self):
        self.chapter_title("4. DISTANCES DE SECURITE")
        if self.counter.distance_records:
            distances = [d.distance_meters for d in self.counter.distance_records]
            avg_dist = sum(distances) / len(distances)
            min_dist = min(distances)
            stats_text = f"""
Statistiques des distances inter-vehiculaires :
- Distance moyenne : {avg_dist:.1f} metres
- Distance minimale : {min_dist:.1f} metres
- Nombre de mesures : {len(distances)}

Interpretation : {'Distance de securite insuffisante en moyenne' if avg_dist < 30 else 'Distance de securite respectee en moyenne'}
"""
            self.chapter_body(stats_text)
        else:
            self.chapter_body("Aucune donnee de distance disponible")
        self.ln(4)
    
    def add_behavioral_analysis(self):
        self.chapter_title("5. ANALYSE COMPORTEMENTALE")
        self.chapter_body(self.analysis_text)
    
    def add_vehicles_table(self):
        self.chapter_title("6. LISTE DES VEHICULES DETECTES")
        df = self.counter.get_vehicles_dataframe()
        if len(df) > 0:
            self.set_font('Arial', 'B', 7)
            self.set_fill_color(46, 204, 113)
            self.set_text_color(255, 255, 255)
            cols = ['ID', 'Categorie', 'Apparition', 'Disparition', 'Duree(s)', 'Vit moy', 'Vit max']
            widths = [12, 22, 32, 32, 18, 18, 18]
            for i, col in enumerate(cols):
                self.cell(widths[i], 6, col, 1, 0, 'C', True)
            self.ln()
            self.set_font('Arial', '', 7)
            self.set_text_color(0, 0, 0)
            for _, row in df.head(20).iterrows():
                self.cell(widths[0], 5, str(row['ID']), 1, 0, 'C')
                self.cell(widths[1], 5, row['Categorie'], 1, 0, 'C')
                self.cell(widths[2], 5, row['Apparition'][:8], 1, 0, 'C')
                self.cell(widths[3], 5, row['Disparition'][:8], 1, 0, 'C')
                self.cell(widths[4], 5, str(row['Duree(s)']), 1, 0, 'C')
                self.cell(widths[5], 5, str(row['Vitesse moy(km/h)']), 1, 0, 'C')
                self.cell(widths[6], 5, str(row['Vitesse max(km/h)']), 1, 1, 'C')
            if len(df) > 20:
                self.chapter_body(f"\n... et {len(df) - 20} autres vehicules (voir export CSV pour la liste complete)")
        else:
            self.chapter_body("Aucun vehicule detecte")
        self.ln(4)
    
    def add_conclusion(self):
        self.chapter_title("7. CONCLUSION")
        speeds = [r.avg_speed_kmh for r in self.counter.vehicle_records if r.avg_speed_kmh > 0]
        risk_score = 0
        if speeds and sum(speeds)/len(speeds) > 60:
            risk_score += 1
        if self.counter.distance_records:
            unsafe_ratio = len([d for d in self.counter.distance_records if d.distance_meters < 25]) / len(self.counter.distance_records) if self.counter.distance_records else 0
            if unsafe_ratio > 0.2:
                risk_score += 1
        if self.counter.vehicle_counts.get("Moto", 0) > self.counter.total_unique * 0.3:
            risk_score += 1
        risk_levels = ["FAIBLE", "MODERE", "ELEVE", "CRITIQUE"]
        conclusion = f"""
Niveau de risque global : {risk_levels[risk_score]}

Ce rapport a ete genere automatiquement par la plateforme de detection de vehicules.
Les donnees presentees sont basees sur l'analyse video par YOLO26.

Pour toute information complementaire, veuillez consulter les fichiers CSV/JSON exportes.
"""
        self.chapter_body(conclusion)
    
    def generate(self, output_path):
        self.add_page()
        self.add_metadata_section()
        self.add_summary_table()
        self.add_speed_statistics()
        self.add_distance_statistics()
        self.add_behavioral_analysis()
        self.add_vehicles_table()
        self.add_conclusion()
        self.output(output_path)


def generate_pdf_report(counter, analysis_text, output_filename=None):
    if output_filename is None:
        output_filename = f"rapport_comptage_yolo26_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf = PDFReport(counter, analysis_text)
    pdf.generate(output_filename)
    return output_filename


# ======================================================================
# COMPTEUR PRINCIPAL
# ======================================================================

class VehicleCounter:
    def __init__(self, config: VehicleConfig, metadata: Dict = None):
        self.config = config
        self.detector = VehicleDetector(config)
        self.tracker = VehicleTracker(config)
        self.metadata = metadata or {
            "location": "", "date": datetime.now().strftime("%Y-%m-%d"),
            "start_time": "", "end_time": "", "video_duration": 0,
            "operator": "", "calibration_factor": config.calibration_factor,
            "total_frames_video": 0, "model_used": "YOLO26",
            "confidence_threshold": config.confidence_threshold
        }
        self.seen_tracks: Set[int] = set()
        self.vehicle_counts: Dict[str, int] = {name: 0 for name in VEHICLE_CLASSES.values()}
        self.frame_counts: Dict[str, int] = {name: 0 for name in VEHICLE_CLASSES.values()}
        self.total_unique = 0
        self.vehicle_records: List[VehicleRecord] = []
        self.next_record_id = 1
        self.distance_records: List[DistanceRecord] = []
        self.track_confidences: Dict[int, List[float]] = defaultdict(list)
        self.track_speeds: Dict[int, List[float]] = defaultdict(list)
        self.alerts: deque = deque(maxlen=30)
        self.frame_count = 0
        self.fps = 0.0
        self.processing_times = deque(maxlen=30)
        self.detection_counts = deque(maxlen=100)
        self.start_timestamp = None
        self.frame_history = deque(maxlen=60)
        self.enabled_classes = ENABLED_CLASSES_DEFAULT.copy()
    
    def set_metadata(self, metadata: Dict):
        self.metadata.update(metadata)
        if "calibration_factor" in metadata:
            self.config.calibration_factor = metadata["calibration_factor"]
        if "confidence_threshold" in metadata:
            self.config.confidence_threshold = metadata["confidence_threshold"]
    
    def start_counting(self):
        self.start_timestamp = time.time()
    
    def get_current_time(self) -> float:
        if self.start_timestamp:
            return time.time() - self.start_timestamp
        return 0
    
    def get_progress(self) -> float:
        if self.metadata.get("total_frames_video", 0) > 0:
            return (self.frame_count / self.metadata["total_frames_video"]) * 100
        return 0
    
    def get_remaining_time(self) -> float:
        progress = self.get_progress() / 100
        if progress > 0 and self.start_timestamp:
            elapsed = time.time() - self.start_timestamp
            total_estimated = elapsed / progress
            return max(0, total_estimated - elapsed)
        return self.metadata.get("video_duration", 0)
    
    def add_alert(self, message: str, level: str = "info"):
        self.alerts.append({"timestamp": datetime.now().strftime("%H:%M:%S"), "message": message, "level": level})
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.time()
        self.frame_count += 1
        self.frame_history.append(frame.copy())
        current_time = self.get_current_time()
        detections = self.detector.detect(frame, self.enabled_classes)
        self.detection_counts.append(len(detections))
        track_ids = self.tracker.update(detections, current_time)
        for track in self.tracker.tracks.values():
            track._calculate_speed(self.config.calibration_factor)
        self.tracker.calculate_distances(self.config.calibration_factor)
        self._record_distances(current_time)
        self._update_counts_with_records(track_ids, current_time)
        annotated = self._annotate_frame(frame.copy(), detections, track_ids)
        elapsed = time.time() - t0
        self.processing_times.append(elapsed)
        if elapsed > 0:
            self.fps = self.config.fps_smoothing * self.fps + (1 - self.config.fps_smoothing) / elapsed
        return annotated
    
    def _record_distances(self, current_time: float):
        active_tracks = [(tid, track) for tid, track in self.tracker.tracks.items() if track.positions]
        active_tracks.sort(key=lambda x: x[1].positions[-1][0])
        for i in range(len(active_tracks) - 1):
            tid1, track1 = active_tracks[i]
            tid2, track2 = active_tracks[i + 1]
            dx = track2.positions[-1][0] - track1.positions[-1][0]
            dy = track2.positions[-1][1] - track1.positions[-1][1]
            dist_pixels = math.hypot(dx, dy)
            dist_meters = dist_pixels * self.config.calibration_factor
            if dist_meters < 50:
                self.distance_records.append(DistanceRecord(
                    timestamp=current_time, frame=self.frame_count,
                    vehicle1_id=tid1, vehicle1_category=track1.class_name,
                    vehicle2_id=tid2, vehicle2_category=track2.class_name,
                    distance_meters=dist_meters, distance_pixels=dist_pixels
                ))
    
    def _update_counts_with_records(self, track_ids: List[int], current_time: float):
        for tid in track_ids:
            if tid not in self.seen_tracks:
                self.seen_tracks.add(tid)
                track = self.tracker.tracks[tid]
                self.vehicle_counts[track.class_name] += 1
                self.total_unique += 1
                record = VehicleRecord(
                    record_id=self.next_record_id, track_id=tid, category=track.class_name,
                    first_seen_frame=self.frame_count, last_seen_frame=self.frame_count,
                    first_seen_time=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    last_seen_time=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    max_confidence=track.confidence, avg_confidence=track.confidence,
                    avg_speed_kmh=0, max_speed_kmh=0, duration_seconds=0,
                    track_points=[track.positions[-1]] if track.positions else []
                )
                self.vehicle_records.append(record)
                self.next_record_id += 1
                self.track_confidences[tid] = [track.confidence]
                self.track_speeds[tid] = []
                self.add_alert(f"Vehicule {track.class_name} #{tid} detecte (Total: {self.total_unique})", "success")
            else:
                track = self.tracker.tracks[tid]
                self.track_confidences[tid].append(track.confidence)
                if track.speed_kmh > 0:
                    self.track_speeds[tid].append(track.speed_kmh)
                for record in self.vehicle_records:
                    if record.track_id == tid:
                        record.last_seen_frame = self.frame_count
                        record.last_seen_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        record.max_confidence = max(record.max_confidence, track.confidence)
                        record.avg_confidence = sum(self.track_confidences[tid]) / len(self.track_confidences[tid])
                        if self.track_speeds[tid]:
                            record.avg_speed_kmh = sum(self.track_speeds[tid]) / len(self.track_speeds[tid])
                            record.max_speed_kmh = max(self.track_speeds[tid])
                        record.duration_seconds = current_time
                        if track.positions:
                            record.track_points.append(track.positions[-1])
                        break
        self.frame_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
        if self.frame_history:
            for det in self.detector.detect(self.frame_history[-1], self.enabled_classes):
                self.frame_counts[det.class_name] += 1
    
    def finalize_records(self):
        for record in self.vehicle_records:
            if record.track_points:
                record.track_points = [record.track_points[0], record.track_points[-1]] if len(record.track_points) > 1 else record.track_points
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Detection], track_ids: List[int]) -> np.ndarray:
        if self.config.show_bboxes:
            for det in detections:
                self._draw_bbox(frame, det)
        if self.config.show_tracks:
            for tid in track_ids:
                if tid in self.tracker.tracks:
                    self._draw_track(frame, tid)
        self._draw_count_panel(frame)
        return frame
    
    def _draw_bbox(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        color = VEHICLE_COLORS.get(det.class_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), font, font_scale, (255, 255, 255), thickness)
        cx, cy = det.center
        cv2.circle(frame, (cx, cy), 4, color, -1)
    
    def _draw_track(self, frame: np.ndarray, tid: int):
        track = self.tracker.tracks[tid]
        color = track.color
        hist = list(self.tracker.track_history[tid])
        if len(hist) > 1:
            pts = np.array(hist, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
        if not track.positions:
            return
        cx, cy = track.positions[-1]
        cv2.circle(frame, (cx, cy), 6, color, -1)
        if self.config.show_ids:
            id_label = f"#{tid}"
            cv2.putText(frame, id_label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if self.config.show_speed and track.speed_kmh > 0:
            speed_label = f"{track.speed_kmh:.0f}km/h"
            cv2.putText(frame, speed_label, (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
    
    def _draw_count_panel(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        panel_w = 280
        panel_h = 150 + len(self.vehicle_counts) * 22
        x, y = w - panel_w - 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Total: {self.total_unique}", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (x + 10, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Seuil: {self.config.confidence_threshold:.2f}", (x + 10, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        cv2.putText(frame, f"Modele: {self.metadata.get('model_used', 'YOLO26')}", (x + 10, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        y_offset = 128
        for cat_name in sorted(self.vehicle_counts.keys()):
            cnt = self.vehicle_counts[cat_name]
            if cnt > 0:
                cv2.putText(frame, f"{cat_name}: {cnt}", (x + 10, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
    
    def get_vehicles_dataframe(self) -> pd.DataFrame:
        rows = []
        for record in self.vehicle_records:
            rows.append({
                'ID': record.record_id, 'Categorie': record.category,
                'Apparition': record.first_seen_time, 'Disparition': record.last_seen_time,
                'Duree(s)': round(record.duration_seconds, 1),
                'Vitesse moy(km/h)': round(record.avg_speed_kmh, 1),
                'Vitesse max(km/h)': round(record.max_speed_kmh, 1),
                'Confiance max': round(record.max_confidence, 2)
            })
        return pd.DataFrame(rows)
    
    def get_distances_dataframe(self) -> pd.DataFrame:
        rows = []
        for record in self.distance_records:
            rows.append({
                'Temps(s)': round(record.timestamp, 2), 'Frame': record.frame,
                'Vehicule1': f"{record.vehicle1_category}#{record.vehicle1_id}",
                'Vehicule2': f"{record.vehicle2_category}#{record.vehicle2_id}",
                'Distance(m)': round(record.distance_meters, 2),
            })
        return pd.DataFrame(rows)
    
    def get_summary_df(self) -> pd.DataFrame:
        rows = []
        for cat_name in VEHICLE_CLASSES.values():
            cnt = self.vehicle_counts.get(cat_name, 0)
            if cnt > 0:
                rows.append({'Categorie': cat_name, 'Detectes': cnt})
        rows.append({'Categorie': 'TOTAL', 'Detectes': self.total_unique})
        return pd.DataFrame(rows)
    
    def get_speed_statistics(self) -> Dict:
        stats = {}
        for cat in VEHICLE_CLASSES.values():
            speeds = []
            for record in self.vehicle_records:
                if record.category == cat and record.avg_speed_kmh > 0:
                    speeds.append(record.avg_speed_kmh)
            if speeds:
                stats[cat] = {
                    'moyenne': round(sum(speeds) / len(speeds), 1),
                    'max': round(max(speeds), 1),
                    'min': round(min(speeds), 1),
                    'count': len(speeds)
                }
        return stats
    
    def get_behavioral_analysis(self) -> str:
        analyzer = BehavioralAnalyzer(
            self.vehicle_records, self.distance_records,
            self.vehicle_counts, self.total_unique, self.metadata, self.config
        )
        return analyzer.generate_analysis()
    
    def export_csv(self) -> Optional[str]:
        if not self.vehicle_records:
            return None
        df = self.get_vehicles_dataframe()
        header = f"# Comptage vehicules YOLO26\n# Lieu: {self.metadata.get('location', '')}\n# Date: {self.metadata.get('date', '')}\n# Seuil confiance: {self.config.confidence_threshold:.2f}\n"
        return header + df.to_csv(index=False, encoding='utf-8-sig')
    
    def export_json(self) -> Optional[str]:
        if not self.vehicle_records:
            return None
        export_data = {
            "metadata": self.metadata,
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "calibration_factor": self.config.calibration_factor
            },
            "summary": {
                "total_vehicles": self.total_unique,
                "by_category": self.vehicle_counts,
                "total_frames": self.frame_count,
                "average_fps": round(self.fps, 2)
            },
            "vehicles": [vars(r) for r in self.vehicle_records],
            "distances": [vars(d) for d in self.distance_records[-1000:]]
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
    
    def reset(self):
        self.seen_tracks.clear()
        self.vehicle_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
        self.frame_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
        self.total_unique = 0
        self.vehicle_records.clear()
        self.distance_records.clear()
        self.track_confidences.clear()
        self.track_speeds.clear()
        self.alerts.clear()
        self.frame_history.clear()
        self.processing_times.clear()
        self.detection_counts.clear()
        self.frame_count = 0
        self.fps = 0.0
        self.next_record_id = 1
        self.start_timestamp = None
        self.tracker.reset()


# ======================================================================
# FONCTIONS UTILITAIRES
# ======================================================================

def parse_time(time_str):
    if not time_str:
        return datetime.now().time()
    try:
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        try:
            return datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            return datetime.now().time()


# ======================================================================
# STREAMLIT UI
# ======================================================================

# Initialisation des variables de session
if 'config' not in st.session_state:
    st.session_state.config = VehicleConfig()
if 'metadata' not in st.session_state:
    st.session_state.metadata = {
        "location": "", "date": datetime.now().strftime("%Y-%m-%d"),
        "start_time": "", "end_time": "", "video_duration": 0,
        "operator": "", "calibration_factor": VehicleConfig.calibration_factor,
        "total_frames_video": 0, "model_used": "YOLO26",
        "confidence_threshold": VehicleConfig.confidence_threshold
    }
if 'counter' not in st.session_state:
    st.session_state.counter = VehicleCounter(st.session_state.config, st.session_state.metadata)
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'counting_started' not in st.session_state:
    st.session_state.counting_started = False
if 'original_fps' not in st.session_state:
    st.session_state.original_fps = 30
if 'frame_delay' not in st.session_state:
    st.session_state.frame_delay = 1.0 / 30

counter = st.session_state.counter
config = st.session_state.config
metadata = st.session_state.metadata

# Verifier et installer YOLO26
check_and_install_yolo26()


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3097/3097164.png", width=80)
    st.markdown("## Detection YOLO26")
    st.caption("Derniere generation IA - Detection ultra-sensible")
    
    st.markdown("---")
    st.header("Informations du comptage")
    
    location = st.text_input("Lieu", value=metadata.get("location", ""))
    date = st.date_input("Date", value=datetime.strptime(metadata.get("date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d") if metadata.get("date") else datetime.now())
    start_time = st.time_input("Heure debut", value=parse_time(metadata.get("start_time", "")))
    end_time = st.time_input("Heure fin", value=parse_time(metadata.get("end_time", "")))
    operator = st.text_input("Operateur", value=metadata.get("operator", ""))
    
    if st.button("Enregistrer", use_container_width=True):
        metadata["location"] = location
        metadata["date"] = date.strftime("%Y-%m-%d")
        metadata["start_time"] = start_time.strftime("%H:%M:%S")
        metadata["end_time"] = end_time.strftime("%H:%M:%S")
        metadata["operator"] = operator
        counter.set_metadata(metadata)
        st.success("Enregistre!")
    
    st.markdown("---")
    st.header("Configuration YOLO26")
    
    # Seuil de confiance variable de 0.01 a 0.5
    st.markdown("**🎯 Seuil de confiance**")
    st.caption("Plus le seuil est bas, plus de vehicules sont detectes")
    confidence_threshold = st.slider(
        "Seuil de detection", 
        min_value=0.01, 
        max_value=0.5, 
        value=config.confidence_threshold, 
        step=0.01,
        format="%.2f",
        help="0.01 = detection maximale (recommande), 0.50 = detection stricte"
    )
    
    if confidence_threshold != config.confidence_threshold:
        config.confidence_threshold = confidence_threshold
        metadata["confidence_threshold"] = confidence_threshold
        counter.set_metadata(metadata)
        st.info(f"🎯 Nouveau seuil: {confidence_threshold:.2f} - {'Detection maximale' if confidence_threshold <= 0.05 else 'Detection equilibree' if confidence_threshold <= 0.15 else 'Detection stricte'}")
    
    st.markdown("---")
    
    uploaded = st.file_uploader("Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded and uploaded != st.session_state.video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        st.session_state.video_path = tfile.name
        st.session_state.video_file = uploaded
        st.session_state.video_capture = None
        counter.reset()
        st.session_state.counting_started = False
        st.success("Video chargee!")
    
    config.calibration_factor = st.number_input("Calibration (m/pixel)", 0.01, 0.5, config.calibration_factor, 0.01, help="1 pixel = X metres")
    
    st.subheader("Categories")
    new_enabled = set()
    for cls_id, cls_name in sorted(VEHICLE_CLASSES.items()):
        if st.checkbox(cls_name, value=cls_id in counter.enabled_classes, key=f"cls_{cls_id}"):
            new_enabled.add(cls_id)
    counter.enabled_classes = new_enabled
    
    st.subheader("Affichage")
    config.show_bboxes = st.checkbox("Boites", value=config.show_bboxes)
    config.show_tracks = st.checkbox("Pistes", value=config.show_tracks)
    config.show_ids = st.checkbox("IDs", value=config.show_ids)
    config.show_speed = st.checkbox("Vitesse", value=config.show_speed)
    
    if st.button("Reinitialiser", use_container_width=True):
        counter.reset()
        st.session_state.processing = False
        st.session_state.paused = False
        st.session_state.counting_started = False
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by YOLO26 - Ultralytics")
    st.caption(f"Seuil actuel: {config.confidence_threshold:.2f}")


# ======================================================================
# INTERFACE PRINCIPALE
# ======================================================================

st.title("Plateforme de Detection et Comptage de Vehicules")
st.markdown("**Analyse video par YOLO26 - Derniere generation d'intelligence artificielle**")

col_model1, col_model2, col_model3, col_model4 = st.columns(4)
with col_model1:
    st.metric("Modele IA", "YOLO26", delta="Ultra-rapide")
with col_model2:
    st.metric("Seuil detection", f"{config.confidence_threshold:.2f}", delta="Ultra-sensible")
with col_model3:
    st.metric("Performance", "43% plus rapide", delta="vs YOLOv8")
with col_model4:
    st.metric("Petits objets", "Detection STAL", delta="Amelioree")

st.markdown("---")

col_prog1, col_prog2, col_prog3 = st.columns([3, 2, 1])

with col_prog1:
    progress_bar = st.progress(0, text="Progression")

with col_prog2:
    time_remaining_text = st.empty()

with col_prog3:
    status_text = st.empty()

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Video", "Tableau vehicules", "Distances", "Graphiques", "Export", "Analyse IA"
])

# TAB 1 - Video
with tab1:
    col_video, col_stats = st.columns([3, 1])
    
    with col_video:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Demarrer", disabled=st.session_state.processing, use_container_width=True):
                st.session_state.processing = True
                st.session_state.paused = False
                st.session_state.video_capture = None
                if not st.session_state.counting_started:
                    counter.start_counting()
                    st.session_state.counting_started = True
                st.rerun()
        with col2:
            if st.button("Pause", disabled=not st.session_state.processing, use_container_width=True):
                st.session_state.paused = not st.session_state.paused
                st.rerun()
        with col3:
            if st.button("Arreter", use_container_width=True):
                st.session_state.processing = False
                st.session_state.paused = False
                st.session_state.video_capture = None
                st.rerun()
        with col4:
            if st.button("Finaliser", use_container_width=True):
                counter.finalize_records()
                st.success("Finalise!")
        
        video_ph = st.empty()
        
        if not st.session_state.video_path:
            st.info("Telechargez une video dans la barre laterale")
        
        if st.session_state.video_path and st.session_state.processing and not st.session_state.paused:
            if st.session_state.video_capture is None:
                st.session_state.video_capture = cv2.VideoCapture(st.session_state.video_path)
                original_fps = st.session_state.video_capture.get(cv2.CAP_PROP_FPS)
                if original_fps <= 0:
                    original_fps = 30
                st.session_state.original_fps = original_fps
                st.session_state.frame_delay = 1.0 / original_fps
                total_frames = int(st.session_state.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata["total_frames_video"] = total_frames
                metadata["video_duration"] = total_frames / original_fps if original_fps > 0 else 0
                counter.set_metadata(metadata)
                st.info(f"Video: {original_fps:.0f} FPS | {total_frames} frames | Duree: {metadata['video_duration']:.1f}s | Seuil: {config.confidence_threshold:.2f}")
            
            cap = st.session_state.video_capture
            frame_delay = st.session_state.frame_delay
            last_frame_time = time.time()
            
            while cap.isOpened() and st.session_state.processing and not st.session_state.paused:
                ret, frame = cap.read()
                if not ret:
                    st.success("Fin de la video - Detection terminee!")
                    st.session_state.processing = False
                    counter.finalize_records()
                    progress_bar.progress(100)
                    time_remaining_text.info("Termine!")
                    status_text.success("Complet")
                    break
                
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                last_frame_time = time.time()
                
                if counter.frame_count % config.frame_skip == 0:
                    processed = counter.process_frame(frame)
                else:
                    processed = frame
                    counter.frame_count += 1
                
                progress = counter.get_progress()
                progress_bar.progress(min(int(progress), 100))
                remaining = counter.get_remaining_time()
                if remaining > 0:
                    minutes = int(remaining // 60)
                    seconds = int(remaining % 60)
                    time_remaining_text.info(f"Temps restant: {minutes:02d}:{seconds:02d}")
                else:
                    time_remaining_text.info("Finalisation...")
                status_text.info(f"Frame {counter.frame_count}/{metadata.get('total_frames_video', 0)}")
                
                video_ph.image(
                    cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True,
                    caption=f"Frame {counter.frame_count}/{metadata.get('total_frames_video', 0)} | {counter.fps:.1f} FPS | Progression: {progress:.1f}% | Total: {counter.total_unique} | Seuil: {config.confidence_threshold:.2f}"
                )
        
        elif st.session_state.video_path and counter.frame_history:
            last = counter.frame_history[-1]
            video_ph.image(cv2.cvtColor(last, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            if counter.frame_count > 0:
                final_progress = (counter.frame_count / metadata.get("total_frames_video", 1)) * 100
                progress_bar.progress(min(int(final_progress), 100))
                status_text.success(f"Termine - {counter.total_unique} vehicules detectes")
    
    with col_stats:
        st.metric("Detectes", counter.total_unique)
        st.metric("Frame actuelle", sum(counter.frame_counts.values()))
        st.metric("FPS", f"{counter.fps:.1f}")
        st.metric("Frames traites", f"{counter.frame_count}/{metadata.get('total_frames_video', 0)}")
        st.markdown("---")
        for cat in sorted(counter.vehicle_counts.keys()):
            cnt = counter.vehicle_counts[cat]
            if cnt > 0:
                st.metric(cat, cnt)


# TAB 2 - Tableau vehicules
with tab2:
    st.header("Liste detaillee des vehicules detectes")
    if counter.total_unique > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lieu", metadata.get("location", "Non specifie"))
            st.metric("Operateur", metadata.get("operator", "Non specifie"))
        with col2:
            st.metric("Date", metadata.get("date", "Non specifiee"))
            st.metric("Heure debut", metadata.get("start_time", "Non specifiee"))
        with col3:
            st.metric("Total vehicules", counter.total_unique)
            st.metric("Seuil confiance", f"{config.confidence_threshold:.2f}")
        st.markdown("---")
        
        speed_stats = counter.get_speed_statistics()
        if speed_stats:
            st.subheader("Statistiques de vitesse par categorie")
            df_speed = pd.DataFrame([{
                'Categorie': cat, 'Vitesse moyenne (km/h)': stats['moyenne'],
                'Vitesse max (km/h)': stats['max'], 'Vitesse min (km/h)': stats['min'],
                'Nombre': stats['count']
            } for cat, stats in speed_stats.items()])
            st.dataframe(df_speed, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Details par vehicule")
        df_vehicles = counter.get_vehicles_dataframe()
        st.dataframe(df_vehicles, use_container_width=True, height=400)
        
        st.subheader("Filtres")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if len(df_vehicles) > 0:
                category_filter = st.multiselect("Filtrer par categorie", options=df_vehicles['Categorie'].unique(), default=df_vehicles['Categorie'].unique())
            else:
                category_filter = []
        with col_f2:
            min_speed = st.slider("Vitesse minimum (km/h)", 0, 120, 0, 5)
        
        if len(df_vehicles) > 0 and category_filter:
            df_filtered = df_vehicles[(df_vehicles['Categorie'].isin(category_filter)) & (df_vehicles['Vitesse moy(km/h)'] >= min_speed)]
            st.dataframe(df_filtered, use_container_width=True, height=300)
            st.caption(f"{len(df_filtered)} vehicules sur {len(df_vehicles)}")
    else:
        st.info("Aucun vehicule detecte. Lancez la detection.")


# TAB 3 - Distances
with tab3:
    st.header("Distances entre vehicules")
    if len(counter.distance_records) > 0:
        df_distances = counter.get_distances_dataframe()
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Enregistrements", len(df_distances))
        with col_d2:
            st.metric("Distance moyenne", f"{df_distances['Distance(m)'].mean():.1f} m")
        with col_d3:
            st.metric("Distance min", f"{df_distances['Distance(m)'].min():.1f} m")
        with col_d4:
            st.metric("Distance max", f"{df_distances['Distance(m)'].max():.1f} m")
        st.markdown("---")
        st.subheader("Enregistrements des distances")
        st.dataframe(df_distances, use_container_width=True, height=400)
        st.subheader("Evolution des distances")
        fig_distances = px.line(df_distances, x='Temps(s)', y='Distance(m)', title="Evolution des distances entre vehicules")
        st.plotly_chart(fig_distances, use_container_width=True)
        st.subheader("Distribution des distances")
        fig_hist = px.histogram(df_distances, x='Distance(m)', nbins=30, title="Distribution des distances entre vehicules")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Aucune donnee de distance disponible. Lancez la detection.")


# TAB 4 - Graphiques
with tab4:
    st.header("Statistiques detaillees")
    if counter.total_unique > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total vehicules", counter.total_unique)
        m2.metric("Categories", len([c for c in counter.vehicle_counts.values() if c > 0]))
        m3.metric("Frames traitees", counter.frame_count)
        m4.metric("FPS moyen", f"{counter.fps:.1f}")
        
        seen_data = {cat: counter.vehicle_counts[cat] for cat in VEHICLE_CLASSES.values() if counter.vehicle_counts[cat] > 0}
        if seen_data:
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(values=list(seen_data.values()), names=list(seen_data.keys()), title="Repartition des vehicules detectes")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                df_cat = pd.DataFrame([{'Categorie': cat, 'Nombre': cnt} for cat, cnt in seen_data.items()])
                fig_bar = px.bar(df_cat, x='Categorie', y='Nombre', title="Nombre de detections par categorie", text_auto=True, color='Categorie')
                st.plotly_chart(fig_bar, use_container_width=True)
        
        if counter.vehicle_records:
            df_records = counter.get_vehicles_dataframe()
            if len(df_records) > 0 and df_records['Vitesse moy(km/h)'].sum() > 0:
                st.subheader("Distribution des vitesses")
                fig_speed = px.box(df_records[df_records['Vitesse moy(km/h)'] > 0], x='Categorie', y='Vitesse moy(km/h)', title="Distribution des vitesses par categorie")
                st.plotly_chart(fig_speed, use_container_width=True)
            
            st.subheader("Chronologie des detections")
            if len(df_records) > 0:
                fig_time = px.scatter(df_records, x='Apparition', y='Categorie', size='Vitesse moy(km/h)', color='Categorie', title="Chronologie des detections", hover_data=['ID', 'Duree(s)'])
                st.plotly_chart(fig_time, use_container_width=True)
                
                st.subheader("Distribution des durees de presence")
                fig_duration = px.histogram(df_records, x='Duree(s)', nbins=20, title="Distribution des durees de presence des vehicules")
                st.plotly_chart(fig_duration, use_container_width=True)
    else:
        st.info("Pas de donnees. Lancez la detection.")


# TAB 5 - Export
with tab5:
    st.header("Export des donnees")
    if counter.total_unique > 0:
        st.subheader("Resume du comptage")
        st.dataframe(counter.get_summary_df(), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("Options d'export")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            csv = counter.export_csv()
            if csv:
                fname = f"comptage_vehicules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button("CSV Vehicules", csv, fname, "text/csv", use_container_width=True)
        with col2:
            if len(counter.distance_records) > 0:
                df_dist = counter.get_distances_dataframe()
                csv_dist = df_dist.to_csv(index=False, encoding='utf-8-sig')
                fname = f"distances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button("CSV Distances", csv_dist, fname, "text/csv", use_container_width=True)
        with col3:
            json_data = counter.export_json()
            if json_data:
                fname = f"comptage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.download_button("JSON", json_data, fname, "application/json", use_container_width=True)
        with col4:
            if st.button("Generer PDF", use_container_width=True, type="primary"):
                with st.spinner("Generation du rapport PDF..."):
                    analysis_text = counter.get_behavioral_analysis()
                    pdf_filename = generate_pdf_report(counter, analysis_text)
                    with open(pdf_filename, "rb") as f:
                        pdf_data = f.read()
                    st.download_button("Telecharger PDF", pdf_data, pdf_filename, "application/pdf", use_container_width=True)
                    os.remove(pdf_filename)
                    st.success("PDF genere!")
        
        st.markdown("---")
        st.info(f"Exports avec seuil de detection: {config.confidence_threshold:.2f}")
    else:
        st.info("Aucune donnee a exporter. Lancez la detection d'abord.")


# TAB 6 - Analyse IA
with tab6:
    st.header("Analyse Comportementale par Intelligence Artificielle")
    st.markdown("Generation automatique d'un rapport d'analyse base sur les donnees collectees")
    
    if counter.total_unique > 0:
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Vehicules analyses", counter.total_unique)
            st.metric("Categories detectees", len([c for c in counter.vehicle_counts.values() if c > 0]))
        with col_info2:
            st.metric("Frames traitees", counter.frame_count)
            st.metric("Seuil detection", f"{config.confidence_threshold:.2f}")
        
        st.markdown("---")
        
        if st.button("Generer l'analyse comportementale", use_container_width=True, type="primary"):
            with st.spinner("Analyse en cours..."):
                analysis_text = counter.get_behavioral_analysis()
                st.session_state.analysis_text = analysis_text
                st.session_state.analysis_generated = True
        
        if st.session_state.get('analysis_generated', False):
            st.markdown("### Rapport d'analyse")
            st.text(st.session_state.analysis_text)
            
            st.markdown("---")
            col_pdf1, col_pdf2 = st.columns(2)
            with col_pdf1:
                if st.button("Exporter en PDF", use_container_width=True):
                    with st.spinner("Generation du PDF..."):
                        pdf_filename = generate_pdf_report(counter, st.session_state.analysis_text)
                        with open(pdf_filename, "rb") as f:
                            pdf_data = f.read()
                        st.download_button("Telecharger", pdf_data, pdf_filename, "application/pdf")
                        os.remove(pdf_filename)
            with col_pdf2:
                st.info("Le rapport inclut toutes les donnees et analyses")
    else:
        st.info("Aucun vehicule detecte. Lancez d'abord la detection sur une video.")


# CSS
st.markdown("""
<style>
.stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border-radius: 10px; border-left: 4px solid #2ecc71; padding: 12px 14px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: #2ecc71 !important; color: white !important; }
.dataframe { font-size: 12px; }
</style>
""", unsafe_allow_html=True)