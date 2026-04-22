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
from io import BytesIO

st.set_page_config(page_title="Comptage Vehicules", page_icon="🚗", layout="wide")


# ======================================================================
# CONFIGURATION DES 8 LIGNES PAR DEFAUT (4 entrees + 4 sorties)
# Coordonnees relatives (0.0 -> 1.0) converties en pixels a la 1ere frame
# ======================================================================
DEFAULT_LINES = [
    # --- 4 ENTREES (vert) ---
    {'name': 'Entree Nord',  'type': 'entry',
     'start': (0.30, 0.07), 'end': (0.70, 0.07)},
    {'name': 'Entree Sud',   'type': 'entry',
     'start': (0.30, 0.93), 'end': (0.70, 0.93)},
    {'name': 'Entree Ouest', 'type': 'entry',
     'start': (0.04, 0.30), 'end': (0.04, 0.70)},
    {'name': 'Entree Est',   'type': 'entry',
     'start': (0.96, 0.30), 'end': (0.96, 0.70)},
    # --- 4 SORTIES (rouge) ---
    {'name': 'Sortie NE',   'type': 'exit',
     'start': (0.55, 0.14), 'end': (0.63, 0.21)},
    {'name': 'Sortie NO',   'type': 'exit',
     'start': (0.16, 0.14), 'end': (0.08, 0.21)},
    {'name': 'Sortie SE',   'type': 'exit',
     'start': (0.63, 0.76), 'end': (0.71, 0.69)},
    {'name': 'Sortie SO',   'type': 'exit',
     'start': (0.12, 0.69), 'end': (0.04, 0.76)},
]

# Couleurs des lignes par type
LINE_COLORS = {
    'entry': (46, 204, 113),    # Vert vif
    'exit':  (231, 76,  60),    # Rouge vif
}


# ======================================================================
# CLASSE PRINCIPALE
# ======================================================================
class VehicleCounter:
    def __init__(self):
        self.setup_model()

        # Classes COCO
        self.vehicle_classes = {
            0: "Personne",
            1: "Velo",
            2: "Voiture",
            3: "Moto",
            5: "Bus",
            7: "Camion",
        }
        self.enabled_classes = {2, 3, 5, 7}

        # Couleurs BGR par categorie
        self.vehicle_colors = {
            "Voiture":  (235, 152,  41),   # bleu-orange
            "Moto":     ( 60,  76, 231),   # rouge
            "Bus":      ( 15, 196, 241),   # jaune
            "Camion":   (173,  68, 142),   # violet
            "Velo":     ( 96, 174,  39),   # vert
            "Personne": (141, 140, 127),   # gris
        }

        # Compteurs globaux et par ligne (franchissements)
        self.category_counts    = {n: 0 for n in self.vehicle_classes.values()}
        self.entry_counts       = {n: 0 for n in self.vehicle_classes.values()}
        self.exit_counts        = {n: 0 for n in self.vehicle_classes.values()}
        self.total_count        = 0
        self.total_entries      = 0
        self.total_exits        = 0
        self.counted_track_ids  = set()   # set de (line_id, track_id)

        # ------------------------------------------------------------------
        # Compteurs de detection BRUTE (independants des lignes)
        #   seen_track_ids       : IDs uniques detectes depuis le debut
        #   seen_category_counts : cumul de vehicules uniques par categorie
        #   seen_total           : total de vehicules uniques detectes
        #   frame_category_counts: vehicules visibles dans la frame courante
        #   frame_total          : total visible dans la frame courante
        # ------------------------------------------------------------------
        self.seen_track_ids        = set()
        self.seen_category_counts  = {n: 0 for n in self.vehicle_classes.values()}
        self.seen_total            = 0
        self.frame_category_counts = {n: 0 for n in self.vehicle_classes.values()}
        self.frame_total           = 0

        # Lignes de comptage
        self.counting_lines = []

        # Suivi
        self.vehicle_tracks = {}
        self.track_history  = defaultdict(lambda: deque(maxlen=50))
        self.track_colors   = {}
        self.next_track_id  = 0

        # Log pour export
        self.detection_log = []

        # Metriques
        self.fps              = 0
        self.frame_count      = 0
        self.processing_times = deque(maxlen=30)
        self.detection_counts = deque(maxlen=100)

        # Affichage
        self.show_tracks = True
        self.show_bboxes = True
        self.show_ids    = True

        # Parametres
        self.confidence_threshold = 0.3
        self.nms_threshold        = 0.4
        self.track_smoothness     = 0.55
        self.frame_skip           = 1

        self.frame_history = deque(maxlen=60)
        self.alerts        = deque(maxlen=30)

    # ------------------------------------------------------------------
    # Modele
    # ------------------------------------------------------------------
    def setup_model(self):
        try:
            from ultralytics import YOLO
            model_path = 'yolov8n.pt'
            with st.spinner("Chargement YOLOv8..."):
                self.model = YOLO(model_path)
            self.use_yolo = True
            st.success("Modele YOLOv8 charge!")
        except Exception as e:
            st.error(f"Erreur modele: {e}")
            self.model    = None
            self.use_yolo = False
            st.warning("Mode simulation active")

    def add_alert(self, message, level="info"):
        self.alerts.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message":   message,
            "level":     level,
        })

    # ------------------------------------------------------------------
    # Ajout d'une ligne (signature compatible avec setup_default_lines)
    # ------------------------------------------------------------------
    def add_line(self, name, line_type, start_point, end_point, direction="auto"):
        """Ajoute une ligne de comptage. line_type: 'entry' ou 'exit'."""
        x1, y1 = start_point
        x2, y2 = end_point
        dx, dy  = x2 - x1, y2 - y1
        angle   = math.degrees(math.atan2(dy, dx))

        if abs(dx) < 1e-6:
            m, b, orient = float('inf'), float(x1), 'vertical'
        else:
            m = dy / dx
            b = y1 - m * x1
            orient = 'horizontal' if abs(angle) < 45 or abs(angle) > 135 else 'vertical'

        color = LINE_COLORS.get(line_type, (52, 152, 219))

        line = {
            'id':              len(self.counting_lines),
            'name':            name,
            'type':            line_type,          # 'entry' | 'exit'
            'start_point':     start_point,
            'end_point':       end_point,
            'm':               m,
            'b':               b,
            'orientation':     orient,
            'count':           0,
            'category_counts': {n: 0 for n in self.vehicle_classes.values()},
            'color':           color,
            'line_width':      4,
            'enabled':         True,
            '_scaled':         True,   # deja en pixels absolus
        }
        self.counting_lines.append(line)
        self.add_alert(f"Ligne '{name}' ({line_type}) ajoutee", "success")
        return line

    def _add_line_relative(self, name, line_type, rel_start, rel_end, fw, fh):
        """Ajoute une ligne a partir de coordonnees relatives (0-1)."""
        sp = (int(rel_start[0] * fw), int(rel_start[1] * fh))
        ep = (int(rel_end[0]   * fw), int(rel_end[1]   * fh))
        return self.add_line(name, line_type, sp, ep)

    # ------------------------------------------------------------------
    # Detection YOLO avec letterbox propre
    # ------------------------------------------------------------------
    def detect_vehicles(self, frame):
        if not self.use_yolo or self.model is None:
            return self.simulate_detections(frame)
        if self.frame_skip > 1 and self.frame_count % self.frame_skip != 0:
            return []
        try:
            oh, ow = frame.shape[:2]
            target = 640
            scale  = min(target / ow, target / oh)
            nw, nh = int(ow * scale), int(oh * scale)
            resized  = cv2.resize(frame, (nw, nh))
            pt = (target - nh) // 2
            pl = (target - nw) // 2
            padded = cv2.copyMakeBorder(
                resized, pt, target-nh-pt, pl, target-nw-pl,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            results = self.model(
                padded, conf=self.confidence_threshold,
                iou=self.nms_threshold, verbose=False,
                classes=list(self.enabled_classes),
            )
            dets = []
            if results and results[0].boxes is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                confs   = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls in zip(boxes, confs, classes):
                    if cls not in self.vehicle_classes or cls not in self.enabled_classes:
                        continue
                    x1 = int(max(0, min(ow, (box[0]-pl)/scale)))
                    y1 = int(max(0, min(oh, (box[1]-pt)/scale)))
                    x2 = int(max(0, min(ow, (box[2]-pl)/scale)))
                    y2 = int(max(0, min(oh, (box[3]-pt)/scale)))
                    bw, bh = x2-x1, y2-y1
                    if bw*bh < 400 or bw < 15 or bh < 15:
                        continue
                    dets.append({
                        'bbox':       [x1, y1, x2, y2],
                        'center':     ((x1+x2)//2, (y1+y2)//2),
                        'confidence': float(conf),
                        'class_id':   int(cls),
                        'class_name': self.vehicle_classes[cls],
                        'width': bw, 'height': bh,
                    })
            self.detection_counts.append(len(dets))
            return dets
        except Exception as e:
            self.add_alert(f"Erreur detection: {e}", "error")
            return self.simulate_detections(frame)

    # ------------------------------------------------------------------
    # Simulation (respecte enabled_classes)
    # ------------------------------------------------------------------
    def simulate_detections(self, frame):
        fh, fw = frame.shape[:2]
        active = list(self.enabled_classes & set(self.vehicle_classes.keys()))
        if not active:
            return []
        np.random.seed(self.frame_count % 1000)
        if not hasattr(self, '_sim_v'):
            self._sim_v = []
        dets, updated = [], []
        for v in self._sim_v:
            if v['class_id'] not in self.enabled_classes:
                continue
            vw, vh = v['width'], v['height']
            dx = int(np.random.randint(-10, 11))
            dy = int(np.random.randint(-6, 7))
            x1 = max(0, min(fw-vw, v['bbox'][0]+dx))
            y1 = max(0, min(fh-vh, v['bbox'][1]+dy))
            v['bbox'] = [x1, y1, x1+vw, y1+vh]
            dets.append({'bbox':[x1,y1,x1+vw,y1+vh],'center':((x1+vw//2),(y1+vh//2)),
                         'confidence':float(np.random.uniform(0.65,0.95)),
                         'class_id':v['class_id'],'class_name':v['class_name'],
                         'width':vw,'height':vh})
            updated.append(v)
        n_new = max(0, np.random.randint(2, 7) - len(updated))
        for _ in range(n_new):
            vw = np.random.randint(70, 200); vh = np.random.randint(45, 120)
            x1 = np.random.randint(0, max(1,fw-vw))
            y1 = np.random.randint(0, max(1,fh-vh))
            cls  = int(np.random.choice(active))
            name = self.vehicle_classes[cls]
            v = {'bbox':[x1,y1,x1+vw,y1+vh],'class_id':cls,'class_name':name,'width':vw,'height':vh}
            updated.append(v)
            dets.append({'bbox':[x1,y1,x1+vw,y1+vh],'center':((x1+vw//2),(y1+vh//2)),
                         'confidence':float(np.random.uniform(0.65,0.95)),
                         'class_id':cls,'class_name':name,'width':vw,'height':vh})
        self._sim_v = updated[:30]
        self.detection_counts.append(len(dets))
        return dets

    # ------------------------------------------------------------------
    # Suivi
    # ------------------------------------------------------------------
    def track_vehicles(self, detections):
        current_ids = []
        if not self.vehicle_tracks:
            for det in detections:
                tid = self.next_track_id
                self._init_track(tid, det)
                current_ids.append(tid)
                self.next_track_id += 1
            return current_ids

        used = set()
        for tid, track in list(self.vehicle_tracks.items()):
            best_i, best_d = -1, float('inf')
            pred = track['positions'][-1]
            for i, det in enumerate(detections):
                if i in used or det['class_name'] != track['class_name']:
                    continue
                d = math.hypot(det['center'][0]-pred[0], det['center'][1]-pred[1])
                if d < best_d:
                    best_d, best_i = d, i
            thr = max(track['width'], track['height']) * 1.8
            if best_i != -1 and best_d < thr:
                self._update_track(tid, detections[best_i])
                used.add(best_i)
                current_ids.append(tid)
            else:
                track['age'] += 1
                if track['age'] > 30:
                    del self.vehicle_tracks[tid]

        for i, det in enumerate(detections):
            if i not in used:
                tid = self.next_track_id
                self._init_track(tid, det)
                current_ids.append(tid)
                self.next_track_id += 1
        return current_ids

    def _init_track(self, tid, det):
        hue   = (tid * 137) % 360
        color = cv2.cvtColor(np.uint8([[[hue, 210, 240]]]), cv2.COLOR_HSV2BGR)[0][0]
        self.track_colors[tid] = tuple(map(int, color))
        self.vehicle_tracks[tid] = {
            'id': tid, 'class_name': det['class_name'], 'class_id': det['class_id'],
            'positions': deque([det['center']], maxlen=60),
            'bboxes':    deque([det['bbox']],   maxlen=20),
            'age': 0, 'first_seen': self.frame_count,
            'color': self.track_colors[tid],
            'width': det['width'], 'height': det['height'],
            'confidence': det['confidence'],
        }
        self.track_history[tid].append(det['center'])

    def _update_track(self, tid, det):
        track = self.vehicle_tracks[tid]
        curr  = det['center']
        if track['positions']:
            last = track['positions'][-1]
            curr = (
                int(self.track_smoothness*last[0] + (1-self.track_smoothness)*curr[0]),
                int(self.track_smoothness*last[1] + (1-self.track_smoothness)*curr[1]),
            )
        track['positions'].append(curr)
        track['bboxes'].append(det['bbox'])
        track['age']        = 0
        track['confidence'] = det['confidence']
        track['width']      = det['width']
        track['height']     = det['height']
        self.track_history[tid].append(curr)

    # ------------------------------------------------------------------
    # Franchissement de ligne
    # ------------------------------------------------------------------
    def count_vehicles(self, track_ids):
        """Compte chaque vehicule une seule fois par ligne franchie."""
        for tid in track_ids:
            if tid not in self.vehicle_tracks:
                continue
            track     = self.vehicle_tracks[tid]
            positions = list(track['positions'])
            if len(positions) < 2:
                continue
            for line in self.counting_lines:
                if not line.get('enabled', True):
                    continue
                key = (line['id'], tid)
                if key in self.counted_track_ids:
                    continue
                crossed = any(
                    self._crosses_line(positions[i], positions[i+1], line)
                    for i in range(len(positions)-1)
                )
                if crossed:
                    self.counted_track_ids.add(key)
                    cat  = track['class_name']
                    ltype = line['type']

                    # Compteurs globaux
                    self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
                    self.total_count += 1
                    if ltype == 'entry':
                        self.entry_counts[cat] = self.entry_counts.get(cat, 0) + 1
                        self.total_entries += 1
                    else:
                        self.exit_counts[cat] = self.exit_counts.get(cat, 0) + 1
                        self.total_exits += 1

                    # Compteur de la ligne
                    line['count'] += 1
                    line['category_counts'][cat] = line['category_counts'].get(cat, 0) + 1

                    # Log
                    self.detection_log.append({
                        'timestamp':   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'frame':       self.frame_count,
                        'track_id':    tid,
                        'categorie':   cat,
                        'ligne':       line['name'],
                        'type_ligne':  ltype,
                        'confiance':   round(track['confidence'], 3),
                        'total_cumul': self.total_count,
                    })
                    sym = "IN" if ltype == 'entry' else "OUT"
                    self.add_alert(
                        f"[{sym}] {cat} #{tid} → {line['name']} | Total: {self.total_count}"
                    )

    # ------------------------------------------------------------------
    # Compteurs de detection brute (sans lignes, sans entree/sortie)
    # ------------------------------------------------------------------
    def _update_seen_counts(self, track_ids, detections):
        """
        Met a jour deux types de compteurs independants des lignes :
          1. frame_category_counts : vehicules visibles dans la frame courante
          2. seen_category_counts  : vehicules uniques detectes depuis le debut
        """
        # --- 1. Comptage frame courante (presence instantanee) ---
        fc = {n: 0 for n in self.vehicle_classes.values()}
        for det in detections:
            cat = det['class_name']
            if cat in fc:
                fc[cat] += 1
        self.frame_category_counts = fc
        self.frame_total = sum(fc.values())

        # --- 2. Comptage cumulatif unique (chaque track compte 1 seule fois) ---
        for tid in track_ids:
            if tid not in self.seen_track_ids and tid in self.vehicle_tracks:
                cat = self.vehicle_tracks[tid]['class_name']
                self.seen_track_ids.add(tid)
                self.seen_category_counts[cat] = self.seen_category_counts.get(cat, 0) + 1
                self.seen_total += 1

    def _crosses_line(self, p1, p2, line):
        x1, y1 = p1; x2, y2 = p2
        if line['orientation'] == 'vertical':
            lx = line['b']
            if (x1 < lx <= x2) or (x2 < lx <= x1):
                if x2 != x1:
                    t  = (lx-x1)/(x2-x1)
                    yi = y1 + t*(y2-y1)
                    ymin = min(line['start_point'][1], line['end_point'][1])
                    ymax = max(line['start_point'][1], line['end_point'][1])
                    return ymin-8 <= yi <= ymax+8
        else:
            m, b = line['m'], line['b']
            d1 = y1-(m*x1+b); d2 = y2-(m*x2+b)
            if d1*d2 < 0 and x2 != x1:
                t  = -d1/(d2-d1)
                xi = x1+t*(x2-x1); yi = y1+t*(y2-y1)
                xmin = min(line['start_point'][0], line['end_point'][0])
                xmax = max(line['start_point'][0], line['end_point'][0])
                ymin = min(line['start_point'][1], line['end_point'][1])
                ymax = max(line['start_point'][1], line['end_point'][1])
                return xmin-8 <= xi <= xmax+8 and ymin-8 <= yi <= ymax+8
        return False

    # ------------------------------------------------------------------
    # Traitement d'une frame
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        t0 = time.time()
        self.frame_count += 1
        self.frame_history.append(frame.copy())

        detections = self.detect_vehicles(frame)
        track_ids  = self.track_vehicles(detections)

        # Mise a jour des compteurs bruts (tous vehicules detectes, sans lignes)
        self._update_seen_counts(track_ids, detections)

        self.count_vehicles(track_ids)

        annotated = self.annotate_frame(frame.copy(), detections, track_ids)

        elapsed = time.time() - t0
        self.processing_times.append(elapsed)
        if elapsed > 0:
            self.fps = 0.8*self.fps + 0.2/elapsed
        if self.frame_count % 200 == 0:
            self._cleanup()
        return annotated

    # ------------------------------------------------------------------
    # ANNOTATION CLAIRE
    # ------------------------------------------------------------------
    def annotate_frame(self, frame, detections, track_ids):
        # 1. Lignes de comptage avec fleche directionnelle
        for line in self.counting_lines:
            if not line.get('enabled', True):
                continue
            self._draw_line(frame, line)

        # 2. Boites de detection (encadrement clair)
        if self.show_bboxes:
            for det in detections:
                self._draw_bbox(frame, det)

        # 3. Pistes
        if self.show_tracks:
            for tid in track_ids:
                if tid in self.vehicle_tracks:
                    self._draw_track(frame, tid)

        # 4. Panneau de comptage
        self._draw_count_panel(frame)
        return frame

    # ---- Dessin d'une ligne de comptage ----
    def _draw_line(self, frame, line):
        sp    = line['start_point']
        ep    = line['end_point']
        color = line['color']
        lw    = line['line_width']

        # Ligne principale epaisse
        cv2.line(frame, sp, ep, color, lw, cv2.LINE_AA)

        # Bordure sombre pour contraste
        cv2.line(frame, sp, ep, (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.line(frame, sp, ep, color,     lw,     cv2.LINE_AA)

        # Petits cercles aux extremites
        cv2.circle(frame, sp, 6, color, -1)
        cv2.circle(frame, ep, 6, color, -1)
        cv2.circle(frame, sp, 7, (255,255,255), 1)
        cv2.circle(frame, ep, 7, (255,255,255), 1)

        # Fleche au milieu indiquant le type
        mx = (sp[0]+ep[0])//2
        my = (sp[1]+ep[1])//2
        dx = ep[0]-sp[0]; dy = ep[1]-sp[1]
        ln = max(math.hypot(dx, dy), 1)
        # Normale a la ligne
        nx = int(-dy/ln * 30); ny = int(dx/ln * 30)
        cv2.arrowedLine(frame, (mx, my), (mx+nx, my+ny), (255,255,255), 2, tipLength=0.5)

        # Label avec fond colore
        ltype = "IN" if line['type'] == 'entry' else "OUT"
        label = f"{line['name']} [{ltype}]: {line['count']}"
        ts    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx    = mx - ts[0]//2
        ty    = my - 18
        # Fond
        cv2.rectangle(frame, (tx-4, ty-ts[1]-4), (tx+ts[0]+4, ty+4), (0,0,0), -1)
        cv2.rectangle(frame, (tx-4, ty-ts[1]-4), (tx+ts[0]+4, ty+4), color, 1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # ---- Encadrement clair des vehicules ----
    def _draw_bbox(self, frame, det):
        x1, y1, x2, y2 = det['bbox']
        color = self.vehicle_colors.get(det['class_name'], (0, 255, 0))
        cat   = det['class_name']
        conf  = det['confidence']

        # --- Bordure externe noire (epaisse) pour contraste ---
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 4)

        # --- Rectangle principal colore ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- Coins en surbrillance (style moderne) ---
        corner_len = max(12, min(20, (x2-x1)//5))
        thickness  = 3
        # Coin haut-gauche
        cv2.line(frame, (x1, y1), (x1+corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1+corner_len), color, thickness)
        # Coin haut-droit
        cv2.line(frame, (x2, y1), (x2-corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1+corner_len), color, thickness)
        # Coin bas-gauche
        cv2.line(frame, (x1, y2), (x1+corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2-corner_len), color, thickness)
        # Coin bas-droit
        cv2.line(frame, (x2, y2), (x2-corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2-corner_len), color, thickness)

        # --- Label : categorie + confiance ---
        label   = f"{cat} {conf:.0%}"
        font    = cv2.FONT_HERSHEY_SIMPLEX
        fscale  = 0.55
        fthick  = 2
        ts      = cv2.getTextSize(label, font, fscale, fthick)[0]
        lx, ly  = x1, y1 - 8

        # Fond du label
        cv2.rectangle(frame,
                      (lx - 1, ly - ts[1] - 6),
                      (lx + ts[0] + 4, ly + 2),
                      (0, 0, 0), -1)
        cv2.rectangle(frame,
                      (lx - 1, ly - ts[1] - 6),
                      (lx + ts[0] + 4, ly + 2),
                      color, 1)
        cv2.putText(frame, label, (lx + 2, ly - 2),
                    font, fscale, (255, 255, 255), fthick, cv2.LINE_AA)

        # Centre du vehicule
        cx, cy = det['center']
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    # ---- Piste de deplacement ----
    def _draw_track(self, frame, tid):
        track = self.vehicle_tracks[tid]
        color = track['color']

        # Trace du chemin
        hist = list(self.track_history[tid])
        if len(hist) > 1:
            pts = np.array(hist, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

        if not track['positions']:
            return

        cx, cy  = track['positions'][-1]
        counted = any((ln['id'], tid) in self.counted_track_ids
                      for ln in self.counting_lines)

        # Cercle de position
        ring_color = (0, 255, 80) if counted else (255, 255, 255)
        cv2.circle(frame, (cx, cy), 8,  color,      -1)
        cv2.circle(frame, (cx, cy), 10, ring_color,  2)

        # ID + categorie
        if self.show_ids:
            id_lbl = f"#{tid} {track['class_name'][:3]}"
            font   = cv2.FONT_HERSHEY_SIMPLEX
            ts     = cv2.getTextSize(id_lbl, font, 0.45, 1)[0]
            bg     = (0, 130, 0) if counted else (30, 30, 30)
            cv2.rectangle(frame,
                          (cx - ts[0]//2 - 3, cy - 32),
                          (cx + ts[0]//2 + 3, cy - 14),
                          bg, -1)
            cv2.rectangle(frame,
                          (cx - ts[0]//2 - 3, cy - 32),
                          (cx + ts[0]//2 + 3, cy - 14),
                          color, 1)
            cv2.putText(frame, id_lbl, (cx - ts[0]//2, cy - 17),
                        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- Panneau de comptage (haut-droite) ----
    def _draw_count_panel(self, frame):
        fh, fw = frame.shape[:2]

        # Categories actives avec valeurs
        entries_cats = [(cat, cnt) for cat, cnt in self.entry_counts.items() if cnt > 0]
        exits_cats   = [(cat, cnt) for cat, cnt in self.exit_counts.items()  if cnt > 0]
        seen_cats    = [(cat, cnt) for cat, cnt in self.seen_category_counts.items() if cnt > 0]
        all_active   = set(
            [c for c,_ in entries_cats] + [c for c,_ in exits_cats] + [c for c,_ in seen_cats]
        )
        n_rows = max(len(all_active), 1)

        panel_w = 310
        panel_h = 115 + n_rows * 26 + 20
        px_     = fw - panel_w - 10
        py_     = 10

        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (px_, py_), (px_+panel_w, py_+panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (px_, py_), (px_+panel_w, py_+panel_h), (200, 200, 200), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Titre
        cv2.putText(frame, "COMPTAGE VEHICULES", (px_+10, py_+22),
                    font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(frame, (px_+8, py_+28), (px_+panel_w-8, py_+28), (70,70,70), 1)

        # Total detections brutes (ligne 1)
        cv2.putText(frame, f"DETECTES : {self.seen_total}",
                    (px_+10, py_+52), font, 0.85, (255, 220, 50), 2, cv2.LINE_AA)
        cv2.line(frame, (px_+8, py_+60), (px_+panel_w-8, py_+60), (60,60,60), 1)

        # Total franchissements lignes (ligne 2)
        entry_c = LINE_COLORS['entry']
        exit_c  = LINE_COLORS['exit']
        cv2.putText(frame, f"IN:{self.total_entries}",
                    (px_+10, py_+80), font, 0.65, entry_c, 2, cv2.LINE_AA)
        cv2.putText(frame, f"OUT:{self.total_exits}",
                    (px_+panel_w//2, py_+80), font, 0.65, exit_c, 2, cv2.LINE_AA)
        cv2.line(frame, (px_+8, py_+88), (px_+panel_w-8, py_+88), (60,60,60), 1)

        # Par categorie : detectes + IN + OUT
        all_cats = sorted(set(
            list(self.seen_category_counts.keys()) +
            [c for c,_ in entries_cats] + [c for c,_ in exits_cats]
        ))
        y = py_ + 108
        max_seen = max(list(self.seen_category_counts.values()) + [1])
        for cat in all_cats:
            seen_cnt = self.seen_category_counts.get(cat, 0)
            e_cnt    = self.entry_counts.get(cat, 0)
            x_cnt    = self.exit_counts.get(cat, 0)
            if seen_cnt == 0 and e_cnt == 0 and x_cnt == 0:
                continue
            color = self.vehicle_colors.get(cat, (200, 200, 200))

            # Barre de proportion (basee sur les detectes)
            bar_w = int((seen_cnt / max_seen) * 95)
            cv2.rectangle(frame, (px_+8, y-15), (px_+8+bar_w, y-4), color, -1)

            cv2.putText(frame, f"{cat:<7}", (px_+10, y-5),
                        font, 0.46, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{seen_cnt}", (px_+105, y-5),
                        font, 0.50, (255, 220, 50), 1, cv2.LINE_AA)
            cv2.putText(frame, f"I:{e_cnt}", (px_+140, y-5),
                        font, 0.44, entry_c, 1, cv2.LINE_AA)
            cv2.putText(frame, f"O:{x_cnt}", (px_+205, y-5),
                        font, 0.44, exit_c, 1, cv2.LINE_AA)
            y += 26

        # FPS
        cv2.putText(frame, f"FPS:{self.fps:.1f}  Frame:{self.frame_count}",
                    (px_+10, py_+panel_h-6),
                    font, 0.37, (120,120,120), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Nettoyage periodique
    # ------------------------------------------------------------------
    def _cleanup(self):
        active = set(self.vehicle_tracks.keys())
        if len(self.counted_track_ids) > 3000:
            old = {k for k in self.counted_track_ids
                   if isinstance(k, tuple) and k[1] not in active}
            keep_n = list(old)[-800:]
            self.counted_track_ids = (
                {k for k in self.counted_track_ids if k not in old} |
                set(keep_n)
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.category_counts       = {n: 0 for n in self.vehicle_classes.values()}
        self.entry_counts          = {n: 0 for n in self.vehicle_classes.values()}
        self.exit_counts           = {n: 0 for n in self.vehicle_classes.values()}
        self.total_count           = 0
        self.total_entries         = 0
        self.total_exits           = 0
        self.counted_track_ids     = set()
        self.seen_track_ids        = set()
        self.seen_category_counts  = {n: 0 for n in self.vehicle_classes.values()}
        self.seen_total            = 0
        self.frame_category_counts = {n: 0 for n in self.vehicle_classes.values()}
        self.frame_total           = 0
        self.vehicle_tracks.clear()
        self.track_history.clear()
        self.track_colors.clear()
        self.next_track_id  = 0
        self.counting_lines = []
        self.detection_log  = []
        self.frame_history.clear()
        self.alerts.clear()
        self.processing_times.clear()
        self.detection_counts.clear()
        self.frame_count = 0
        self.fps         = 0
        if hasattr(self, '_sim_v'):
            self._sim_v = []

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_csv(self):
        if not self.detection_log:
            return None
        return pd.DataFrame(self.detection_log).to_csv(index=False, encoding='utf-8')

    def export_json(self):
        if not self.detection_log:
            return None
        return json.dumps(self.detection_log, indent=2, ensure_ascii=False)

    def get_summary_df(self):
        rows = []
        for cat in self.vehicle_classes.values():
            seen  = self.seen_category_counts.get(cat, 0)
            entry = self.entry_counts.get(cat, 0)
            exit_ = self.exit_counts.get(cat, 0)
            if seen > 0 or entry > 0 or exit_ > 0:
                rows.append({
                    'Categorie':        cat,
                    'Detectes (total)': seen,
                    'Entrees (IN)':     entry,
                    'Sorties (OUT)':    exit_,
                })
        rows.append({
            'Categorie':        'TOTAL',
            'Detectes (total)': self.seen_total,
            'Entrees (IN)':     self.total_entries,
            'Sorties (OUT)':    self.total_exits,
        })
        return pd.DataFrame(rows)


# ======================================================================
# FONCTION SETUP 8 LIGNES PAR DEFAUT
# Compatible avec la signature demandee dans la consigne
# ======================================================================
def setup_default_lines(counter, frame_width=1280, frame_height=720):
    """Configure les 8 lignes de comptage par defaut (4 entrees + 4 sorties)."""
    counter.counting_lines = []   # reset des lignes existantes
    for cfg in DEFAULT_LINES:
        sp = (int(cfg['start'][0] * frame_width),
              int(cfg['start'][1] * frame_height))
        ep = (int(cfg['end'][0]   * frame_width),
              int(cfg['end'][1]   * frame_height))
        counter.add_line(
            name=cfg['name'],
            line_type=cfg['type'],
            start_point=sp,
            end_point=ep,
            direction="auto",
        )
    counter.add_alert("8 lignes de comptage configurees avec succes", "success")
    return len(DEFAULT_LINES)


# ======================================================================
# SESSION STATE
# ======================================================================
if 'counter'        not in st.session_state: st.session_state.counter        = VehicleCounter()
if 'video_file'     not in st.session_state: st.session_state.video_file     = None
if 'video_path'     not in st.session_state: st.session_state.video_path     = None
if 'video_capture'  not in st.session_state: st.session_state.video_capture  = None
if 'processing'     not in st.session_state: st.session_state.processing     = False
if 'paused'         not in st.session_state: st.session_state.paused         = False
if 'lines_setup'    not in st.session_state: st.session_state.lines_setup    = False
if 'frame_wh'       not in st.session_state: st.session_state.frame_wh       = (1280, 720)

counter = st.session_state.counter


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("Configuration")

    # --- Source video ---
    st.subheader("Source Video")
    uploaded = st.file_uploader("Telecharger une video", type=['mp4','avi','mov','mkv'])
    if uploaded and uploaded != st.session_state.video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        # Lire les dimensions reelles de la video
        cap_tmp = cv2.VideoCapture(tfile.name)
        if cap_tmp.isOpened():
            fw = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.session_state.frame_wh = (fw, fh)
            cap_tmp.release()
        st.session_state.video_path    = tfile.name
        st.session_state.video_file    = uploaded
        st.session_state.video_capture = None
        st.session_state.lines_setup   = False
        counter.reset()

    # --- Configuration rapide des lignes ---
    st.subheader("Configuration Rapide")
    if st.button("Configurer 8 lignes par defaut", use_container_width=True, type="primary"):
        fw, fh = st.session_state.frame_wh
        count  = setup_default_lines(counter, fw, fh)
        st.session_state.lines_setup = True
        st.success(f"{count} lignes ajoutees ({fw}x{fh}px)!")
        st.rerun()

    if counter.counting_lines:
        st.caption(f"{len(counter.counting_lines)} ligne(s) configuree(s)")
        n_entry = sum(1 for l in counter.counting_lines if l['type']=='entry')
        n_exit  = sum(1 for l in counter.counting_lines if l['type']=='exit')
        st.caption(f"  IN: {n_entry}  |  OUT: {n_exit}")

    # --- Ajout manuel d'une ligne ---
    with st.expander("Ajouter une ligne manuellement"):
        lname  = st.text_input("Nom", value=f"Ligne {len(counter.counting_lines)+1}")
        ltype  = st.selectbox("Type", ["entry", "exit"],
                              format_func=lambda x: "Entree (IN)" if x=="entry" else "Sortie (OUT)")
        st.markdown("Point A (0.0 - 1.0)")
        ca1, ca2 = st.columns(2)
        ax = ca1.number_input("X1", 0.0, 1.0, 0.05, 0.01, key="ax")
        ay = ca2.number_input("Y1", 0.0, 1.0, 0.5,  0.01, key="ay")
        st.markdown("Point B (0.0 - 1.0)")
        cb1, cb2 = st.columns(2)
        bx = cb1.number_input("X2", 0.0, 1.0, 0.95, 0.01, key="bx")
        by = cb2.number_input("Y2", 0.0, 1.0, 0.5,  0.01, key="by")
        if st.button("Creer", use_container_width=True):
            fw, fh = st.session_state.frame_wh
            sp = (int(ax*fw), int(ay*fh))
            ep = (int(bx*fw), int(by*fh))
            counter.add_line(lname, ltype, sp, ep)
            st.success(f"Ligne '{lname}' creee!")
            st.rerun()

    # --- Liste des lignes ---
    if counter.counting_lines:
        st.subheader("Lignes configurees")
        for i, line in enumerate(counter.counting_lines):
            lbl = "IN" if line['type']=='entry' else "OUT"
            col_hex = "#{:02x}{:02x}{:02x}".format(*line['color'])
            with st.expander(f"{line['name']} [{lbl}] — {line['count']} passages"):
                en = st.checkbox("Active", value=line.get('enabled', True), key=f"en_{i}")
                line['enabled'] = en
                for cat, cnt in line['category_counts'].items():
                    if cnt > 0:
                        st.text(f"  {cat}: {cnt}")
                if st.button("Supprimer", key=f"del_{i}"):
                    counter.counting_lines.pop(i)
                    st.rerun()

    # --- Parametres ---
    st.subheader("Parametres")
    counter.frame_skip            = st.slider("Frame skip", 1, 5, 1)
    counter.confidence_threshold  = st.slider("Seuil confiance", 0.1, 0.9, 0.3, 0.05)

    # --- Categories ---
    st.subheader("Categories")
    new_enabled = set()
    for cls_id, cls_name in sorted(counter.vehicle_classes.items()):
        if st.checkbox(cls_name, value=cls_id in counter.enabled_classes, key=f"cls_{cls_id}"):
            new_enabled.add(cls_id)
    counter.enabled_classes = new_enabled

    # --- Affichage ---
    st.subheader("Affichage")
    counter.show_tracks = st.checkbox("Pistes",  value=counter.show_tracks)
    counter.show_bboxes = st.checkbox("Boites",  value=counter.show_bboxes)
    counter.show_ids    = st.checkbox("IDs",     value=counter.show_ids)

    st.markdown("---")
    if st.button("Reinitialiser tout", use_container_width=True, type="secondary"):
        counter.reset()
        st.session_state.processing    = False
        st.session_state.paused        = False
        st.session_state.video_capture = None
        st.session_state.lines_setup   = False
        st.rerun()


# ======================================================================
# INTERFACE PRINCIPALE
# ======================================================================
st.title("Comptage de Vehicules — 4 Entrees / 4 Sorties")
st.markdown(
    "Detection et comptage par categorie avec **4 lignes d'entree** et **4 lignes de sortie** tracees automatiquement."
)

tab1, tab2, tab3 = st.tabs(["Video & Comptage", "Graphiques", "Export"])

# -----------------------------------------------------------------------
# TAB 1 — Video
# -----------------------------------------------------------------------
with tab1:
    col_vid, col_stats = st.columns([3, 1])

    with col_vid:
        # Alerte si aucune ligne
        if not counter.counting_lines:
            st.warning(
                "Aucune ligne configuree. Cliquez sur **'Configurer 8 lignes par defaut'** dans la barre laterale."
            )

        if st.session_state.video_path:
            bt1, bt2, bt3 = st.columns(3)
            with bt1:
                if st.button("Demarrer", disabled=st.session_state.processing,
                             use_container_width=True):
                    st.session_state.processing    = True
                    st.session_state.paused        = False
                    st.session_state.video_capture = None
            with bt2:
                if st.button("Pause / Reprise",
                             disabled=not st.session_state.processing,
                             use_container_width=True):
                    st.session_state.paused = not st.session_state.paused
            with bt3:
                if st.button("Arreter", use_container_width=True):
                    st.session_state.processing    = False
                    st.session_state.paused        = False
                    st.session_state.video_capture = None

            video_ph = st.empty()

            if st.session_state.processing and not st.session_state.paused:
                if st.session_state.video_capture is None:
                    st.session_state.video_capture = cv2.VideoCapture(
                        st.session_state.video_path
                    )
                cap = st.session_state.video_capture

                while (cap.isOpened()
                       and st.session_state.processing
                       and not st.session_state.paused):
                    ret, frame = cap.read()
                    if not ret:
                        st.info("Fin de la video")
                        st.session_state.processing = False
                        break

                    # Si les lignes n'ont pas encore ete adaptees aux dimensions reelles
                    fh_real, fw_real = frame.shape[:2]
                    st.session_state.frame_wh = (fw_real, fh_real)

                    # Auto-setup des lignes par defaut au premier frame si aucune ligne
                    if not counter.counting_lines:
                        setup_default_lines(counter, fw_real, fh_real)

                    processed = counter.process_frame(frame)
                    video_ph.image(
                        cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True,
                        caption=f"Frame {counter.frame_count} | FPS {counter.fps:.1f} | Total: {counter.total_count}"
                    )
                    time.sleep(0.001)

            elif counter.frame_history:
                last = counter.frame_history[-1]
                video_ph.image(
                    cv2.cvtColor(last, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True,
                    caption=f"Derniere frame: {counter.frame_count}"
                )
        else:
            st.info("Telechargez une video pour commencer.")

    # ---- Colonne compteurs ----
    with col_stats:
        st.markdown("### Compteurs")

        # --- Detectes total (brut, sans lignes) ---
        st.metric("DETECTES (total unique)", counter.seen_total,
                  help="Nombre de vehicules uniques detectes depuis le debut, sans tenir compte des lignes")

        # --- Franchissements lignes ---
        c1, c2 = st.columns(2)
        c1.metric("IN",  counter.total_entries, help="Vehicules ayant franchi une ligne d'entree")
        c2.metric("OUT", counter.total_exits,   help="Vehicules ayant franchi une ligne de sortie")

        # --- Frame courante ---
        st.caption(f"Dans la frame actuelle : {counter.frame_total} vehicule(s)")

        st.markdown("---")
        st.markdown("**Detection par categorie**")
        st.caption("Detectes = vehicules uniques vus | IN / OUT = franchissements")

        enabled_names = [counter.vehicle_classes[c] for c in counter.enabled_classes]
        for cat in counter.vehicle_classes.values():
            if cat not in enabled_names:
                continue
            seen_cnt  = counter.seen_category_counts.get(cat, 0)
            frame_cnt = counter.frame_category_counts.get(cat, 0)
            e_cnt     = counter.entry_counts.get(cat, 0)
            x_cnt     = counter.exit_counts.get(cat, 0)
            if seen_cnt == 0 and frame_cnt == 0:
                continue
            col_hex = "#{:02x}{:02x}{:02x}".format(
                *counter.vehicle_colors.get(cat, (200,200,200)))
            st.markdown(
                f"<div style='padding:6px 10px;margin:3px 0;border-radius:6px;"
                f"border-left:4px solid {col_hex};background:rgba(255,255,255,0.04)'>"
                f"<span style='color:{col_hex};font-weight:bold'>{cat}</span>"
                f"<span style='float:right;font-weight:bold;color:#ffdc32'>{seen_cnt}</span><br>"
                f"<small style='color:#aaa'>"
                f"Frame: <b style='color:white'>{frame_cnt}</b>"
                f" &nbsp;|&nbsp; "
                f"<span style='color:#2ecc71'>IN:{e_cnt}</span>"
                f" &nbsp; "
                f"<span style='color:#e74c3c'>OUT:{x_cnt}</span>"
                f"</small>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.metric("Frames", counter.frame_count)
        st.metric("FPS",    f"{counter.fps:.1f}")
        st.metric("Actifs", len(counter.vehicle_tracks))

        # Activite recente
        if counter.alerts:
            st.markdown("### Activite")
            for alert in list(counter.alerts)[-6:]:
                lc = {"info":"#3498db","warning":"#f39c12",
                      "error":"#e74c3c","success":"#2ecc71"}.get(alert['level'],'#888')
                st.markdown(
                    f"<div style='border-left:3px solid {lc};padding:2px 8px;"
                    f"margin:2px 0;font-size:0.75em'>"
                    f"<span style='color:#aaa'>{alert['timestamp']}</span><br>"
                    f"{alert['message']}</div>",
                    unsafe_allow_html=True
                )

# -----------------------------------------------------------------------
# TAB 2 — Graphiques
# -----------------------------------------------------------------------
with tab2:
    st.header("Analyse graphique")

    if counter.seen_total > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detectes (total)", counter.seen_total)
        m2.metric("Entrees (IN)",     counter.total_entries)
        m3.metric("Sorties (OUT)",    counter.total_exits)
        m4.metric("FPS",              f"{counter.fps:.1f}")

        # Donnees de detection brute
        seen_data = {cat: counter.seen_category_counts[cat]
                     for cat in counter.vehicle_classes.values()
                     if counter.seen_category_counts.get(cat, 0) > 0}

        if seen_data:
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(
                    values=list(seen_data.values()), names=list(seen_data.keys()),
                    title="Repartition des vehicules detectes",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                # Graphique : detectes vs IN vs OUT par categorie
                all_cats = list(seen_data.keys())
                rows_multi = (
                    [{"Categorie": c, "Nombre": seen_data.get(c,0), "Type": "Detectes"} for c in all_cats] +
                    [{"Categorie": c, "Nombre": counter.entry_counts.get(c,0), "Type": "Entree (IN)"} for c in all_cats] +
                    [{"Categorie": c, "Nombre": counter.exit_counts.get(c,0),  "Type": "Sortie (OUT)"} for c in all_cats]
                )
                df_multi = pd.DataFrame(rows_multi)
                fig_multi = px.bar(
                    df_multi, x="Categorie", y="Nombre", color="Type", barmode="group",
                    title="Detectes / Entrees / Sorties par categorie",
                    color_discrete_map={
                        "Detectes":    "#ffdc32",
                        "Entree (IN)": "#2ecc71",
                        "Sortie (OUT)":"#e74c3c",
                    },
                    text_auto=True,
                )
                st.plotly_chart(fig_multi, use_container_width=True)

        # Comptage par ligne
        if counter.counting_lines:
            line_data = [
                {'Ligne': l['name'], 'Type': 'IN' if l['type']=='entry' else 'OUT', 'Total': l['count']}
                for l in counter.counting_lines if l['count'] > 0
            ]
            if line_data:
                fig_lines = px.bar(
                    pd.DataFrame(line_data), x='Ligne', y='Total', color='Type',
                    title="Passages par ligne",
                    color_discrete_map={'IN':'#2ecc71','OUT':'#e74c3c'},
                    text_auto=True,
                )
                st.plotly_chart(fig_lines, use_container_width=True)

        # Courbe d'accumulation
        if counter.detection_log:
            df_log = pd.DataFrame(counter.detection_log)
            df_log['n'] = range(1, len(df_log)+1)
            fig_line = px.line(
                df_log, x='frame', y='n', color='categorie',
                title="Accumulation des detections",
                labels={'frame':'Frame','n':'Cumul','categorie':'Categorie'},
            )
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Lancez l'analyse pour voir les graphiques.")

# -----------------------------------------------------------------------
# TAB 3 — Export
# -----------------------------------------------------------------------
with tab3:
    st.header("Export des donnees")

    if counter.seen_total > 0:
        st.subheader("Resume par categorie")
        st.dataframe(counter.get_summary_df(), use_container_width=True, hide_index=True)

        if counter.detection_log:
            st.subheader("Detail des passages")
            df_log = pd.DataFrame(counter.detection_log)
            st.dataframe(df_log, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                csv_data = counter.export_csv()
                if csv_data:
                    fname = f"comptage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button("Telecharger CSV", csv_data, fname,
                                       "text/csv", use_container_width=True)
            with c2:
                json_data = counter.export_json()
                if json_data:
                    fname = f"comptage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    st.download_button("Telecharger JSON", json_data, fname,
                                       "application/json", use_container_width=True)

        st.markdown("---")
        if st.button("Effacer toutes les donnees", type="secondary"):
            counter.reset()
            st.rerun()
    else:
        st.info("Aucune donnee. Lancez l'analyse d'abord.")

# ======================================================================
# CSS
# ======================================================================
st.markdown("""
<style>
.stButton>button { border-radius:8px; font-weight:500; transition:all .25s; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 3px 10px rgba(0,0,0,0.2); }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    padding: 10px 14px;
}
.stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0; padding:8px 20px; }
.stTabs [aria-selected="true"] { background:#2ecc71 !important; color:white !important; }
</style>
""", unsafe_allow_html=True)
