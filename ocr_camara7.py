import cv2
import numpy as np
import re
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
from collections import deque, Counter

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
CAMARAS = [
    {
        "nombre":          "Cámara 1",
        "index":           0,
        "temp_coords": [177, 74, 358, 174],
        "temp_dec_coords": [373, 115, 442, 173],
        "hum_coords": [179, 197, 444, 281],
    },
    {
        "nombre":          "Cámara 2",
        "index":           1,
        "temp_coords":     [212, 66,  461, 223],
        "temp_dec_coords": [467, 126, 540, 237],
        "hum_coords":      [213, 236, 531, 398],
    },
    {
        "nombre":          "Cámara 3",
        "index":           2,
        "temp_coords":     [212, 66,  461, 223],
        "temp_dec_coords": [467, 126, 540, 237],
        "hum_coords":      [213, 236, 531, 398],
    },
    {
        "nombre":          "Cámara 4",
        "index":           3,
        "temp_coords":     [212, 66,  461, 223],
        "temp_dec_coords": [467, 126, 540, 237],
        "hum_coords":      [213, 236, 531, 398],
    },
]

MODO_CALIBRACION  = False   # True → modo calibración
CAMARA_CALIBRAR   = 0       # índice en CAMARAS a calibrar

DISPLAY_W         = 420
DISPLAY_H         = 315
VOTE_N            = 3       # buffer de votación (reducido para reaccionar más rápido)
INTERVALO_MS      = 1000    # lectura cada segundo
CSV_COMBINADO     = "lecturas_todas_camaras.csv"

# Cambio máximo permitido entre lecturas consecutivas (filtro físico)
MAX_DELTA_TEMP    = 5.0     # °C
MAX_DELTA_HUM     = 10.0    # %


# ═══════════════════════════════════════════════════════════════════════════════
#  MODO CALIBRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
if MODO_CALIBRACION:
    cfg_cal = CAMARAS[CAMARA_CALIBRAR]
    cam_cal = cv2.VideoCapture(cfg_cal["index"])
    if not cam_cal.isOpened():
        print(f"❌ No se pudo abrir {cfg_cal['nombre']} (index={cfg_cal['index']})")
        exit()

    drawing, rect_start, rects = False, (0, 0), []
    ETIQUETAS = ["temp_coords", "temp_dec_coords", "hum_coords"]
    MENSAJES  = [
        "1) Arrastra sobre TEMPERATURA (ej: 26)",
        "2) Arrastra sobre el DECIMAL   (ej: .5)",
        "3) Arrastra sobre HUMEDAD      (ej: 46)",
        "Listo! Presiona Q",
    ]

    def mouse_callback(event, x, y, flags, param):
        global drawing, rect_start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, rect_start = True, (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rects.append((rect_start[0], rect_start[1], x, y))
            r = rects[-1]
            idx = len(rects) - 1
            if idx < len(ETIQUETAS):
                print(f"✅ {ETIQUETAS[idx]} = [{r[0]}, {r[1]}, {r[2]}, {r[3]}]")
            if len(rects) == 3:
                print(f"\n📋 Copia en CAMARAS[{CAMARA_CALIBRAR}]:")
                for i, etiq in enumerate(ETIQUETAS):
                    r2 = rects[i]
                    print(f'    "{etiq}": [{r2[0]}, {r2[1]}, {r2[2]}, {r2[3]}],')
                print("Luego MODO_CALIBRACION = False y corre de nuevo.")

    cv2.namedWindow("CALIBRADOR")
    cv2.setMouseCallback("CALIBRADOR", mouse_callback)
    COLORES = [(0, 255, 0), (0, 200, 255), (255, 100, 0)]
    while True:
        ret, frame = cam_cal.read()
        if not ret:
            break
        disp = frame.copy()
        for i, r in enumerate(rects):
            color = COLORES[i % len(COLORES)]
            cv2.rectangle(disp, (r[0], r[1]), (r[2], r[3]), color, 2)
            cv2.putText(disp, ETIQUETAS[i].replace("_coords","").upper(),
                        (r[0], max(r[1]-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(disp, MENSAJES[min(len(rects), len(MENSAJES)-1)],
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("CALIBRADOR", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam_cal.release()
    cv2.destroyAllWindows()
    exit()


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV
# ═══════════════════════════════════════════════════════════════════════════════
def _init_csvs():
    with open(CSV_COMBINADO, "w", newline="", encoding="utf-8") as f:
        enc = ["Fecha/Hora"]
        for cfg in CAMARAS:
            n = cfg["nombre"]
            enc += [f"{n} – Temperatura (°C)", f"{n} – Humedad (%)"]
        csv.writer(f).writerow(enc)
    for cfg in CAMARAS:
        nombre_archivo = f"lecturas_{cfg['nombre'].replace(' ', '_')}.csv"
        cfg["csv_individual"] = nombre_archivo
        with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "Fecha/Hora",
                f"{cfg['nombre']} – Temperatura (°C)",
                f"{cfg['nombre']} – Humedad (%)",
            ])


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO  –  pipeline CLAHE optimizado para LCD oscuro-sobre-claro
# ═══════════════════════════════════════════════════════════════════════════════
def _corregir_perspectiva(zone):
    """
    Intenta rectificar inclinación leve del display usando los bordes detectados.
    Si no encuentra bordes claros, devuelve la zona sin cambios.
    """
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                            minLineLength=zone.shape[1]//3, maxLineGap=10)
    if lines is None or len(lines) < 2:
        return zone

    # Calcular ángulo promedio de las líneas más horizontales
    angulos = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) < 20:          # solo líneas casi horizontales
            angulos.append(ang)
    if not angulos:
        return zone

    ang_medio = np.median(angulos)
    if abs(ang_medio) < 0.5:      # inclinación despreciable
        return zone

    h, w = zone.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang_medio, 1.0)
    return cv2.warpAffine(zone, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarizar(zone):
    """
    Pipeline mejorado:
      1. Corrección de perspectiva (rotación)
      2. Upscale x2
      3. Desenfoque leve para eliminar ruido de píxeles
      4. CLAHE para realce de contraste local
      5. Prueba múltiples umbrales + adaptativo
      6. Elige el que tiene densidad más cercana a 0.18
    """
    zone = _corregir_perspectiva(zone)
    zone = cv2.resize(zone, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    # Desenfoque leve antes del CLAHE para reducir ruido de sensor
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE: realce de contraste adaptativo local (mejor que equalizeHist global)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    mejores = []
    # Umbrales fijos con rango ampliado para displays oscuros y claros
    for val in [20, 40, 60, 80, 100, 120, 140]:
        _, t = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV)
        ratio = np.sum(t > 0) / t.size
        if 0.03 < ratio < 0.50:
            mejores.append((abs(ratio - 0.18), t))

    # Umbral adaptativo gaussiano
    t_adapt = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 21, 6)
    ratio = np.sum(t_adapt > 0) / t_adapt.size
    if 0.03 < ratio < 0.50:
        mejores.append((abs(ratio - 0.18), t_adapt))

    # Otsu como fallback
    if not mejores:
        _, t_otsu = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return t_otsu

    mejores.sort(key=lambda x: x[0])
    return mejores[0][1]


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE SEGMENTOS LCD
# ═══════════════════════════════════════════════════════════════════════════════
# Mapa (a, f, b, g, e, c, d)
#       a=arriba  f=izq-arriba  b=der-arriba  g=medio
#       e=izq-abajo  c=der-abajo  d=abajo
MAPA_DIGITOS = {
    (1, 1, 1, 0, 1, 1, 1): '0',
    (0, 0, 1, 0, 0, 1, 0): '1',
    (1, 0, 1, 1, 1, 0, 1): '2',
    (1, 0, 1, 1, 0, 1, 1): '3',
    (0, 1, 1, 1, 0, 1, 0): '4',
    (1, 1, 0, 1, 0, 1, 1): '5',
    (1, 1, 0, 1, 1, 1, 1): '6',
    (1, 0, 1, 0, 0, 1, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9',
}


def segmento_activo(thresh, x1, y1, x2, y2, umbral=0.15):
    h, w = thresh.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = thresh[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    return 1 if (np.sum(roi > 0) / roi.size) > umbral else 0


def _segs_con_umbral_diferenciado(thresh, zonas):
    """
    Evalúa los 7 segmentos con umbrales distintos según el segmento:
      - Segmento g (índice 3, el del medio): umbral alto 0.28
        → evita que ruido en la zona central confunda 0→8, 6→8, 5→8
      - Segmentos a/d (índices 0 y 6, horizontales extremos): umbral 0.18
      - Segmentos verticales (f,b,e,c): umbral bajo 0.12
        → los segmentos verticales suelen tener menos píxeles activos
    """
    UMBRALES = [0.18, 0.12, 0.12, 0.28, 0.12, 0.12, 0.18]
    return tuple(
        segmento_activo(thresh, *zonas[i], umbral=UMBRALES[i])
        for i in range(7)
    )


def _zonas_segmento(h, w):
    """
    Divide el recorte de un dígito en 7 zonas de segmento.
    Usa márgenes dinámicos para tolerar segmentos delgados o desalineados.
    """
    m   = max(2, h // 14)    # margen vertical fino
    mw  = max(2, w // 10)    # margen lateral

    mid = h // 2

    # Zonas horizontales (arriba, medio, abajo)
    ya1, ya2 = 0,          h // 6 + m
    yg1, yg2 = mid - m,    mid + m
    yd1, yd2 = h - h//6 - m, h

    # Zonas verticales (izquierda y derecha, mitad superior e inferior)
    yf1, yf2 = m,          mid - m
    yb1, yb2 = m,          mid - m
    ye1, ye2 = mid + m,    h - m
    yc1, yc2 = mid + m,    h - m

    # Franjas horizontales de los segmentos verticales
    xl, xm1, xm2, xr = 0, w // 4, 3 * w // 4, w

    return [
        (xm1,       ya1, xm2,   ya2),   # a – arriba
        (xl,        yf1, w//3,  yf2),   # f – izq-arriba
        (2*w//3,    yb1, xr,    yb2),   # b – der-arriba
        (xm1,       yg1, xm2,   yg2),   # g – medio
        (xl,        ye1, w//3,  ye2),   # e – izq-abajo
        (2*w//3,    yc1, xr,    yc2),   # c – der-abajo
        (xm1,       yd1, xm2,   yd2),   # d – abajo
    ]


def leer_digito(thresh):
    """
    Lee un dígito.
    1. Evalúa segmentos con umbrales diferenciados por zona.
    2. Coincidencia exacta primero.
    3. Hamming ≤ 1 solamente (antes era ≤2, lo que causaba 0→8 y 6→8).
       Con Hamming=2 un '0' con ruido en g se confundía con '8' porque
       solo difieren en 1 segmento; con ≤1 solo acepta si hay 1 error limpio.
    Devuelve (char, segs_tuple, zonas_list).
    """
    h, w = thresh.shape
    zonas = _zonas_segmento(h, w)
    segs  = _segs_con_umbral_diferenciado(thresh, zonas)

    if segs in MAPA_DIGITOS:
        return MAPA_DIGITOS[segs], segs, zonas

    # Hamming ≤ 1 con desempate estricto
    candidatos = []
    for patron, char in MAPA_DIGITOS.items():
        dist = sum(a != b for a, b in zip(segs, patron))
        candidatos.append((dist, char))
    candidatos.sort(key=lambda x: x[0])

    mejor_dist = candidatos[0][0]
    if mejor_dist == 1:
        empates = [c for c in candidatos if c[0] == 1]
        if len(empates) == 1:
            return empates[0][1], segs, zonas

    return '?', segs, zonas


# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN DE DÍGITOS
# ═══════════════════════════════════════════════════════════════════════════════
def separar_digitos(thresh):
    altura = thresh.shape[0]
    # Dilatar verticalmente para conectar segmentos del mismo dígito
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dil = cv2.dilate(thresh, kv, iterations=4)
    dil = cv2.dilate(dil,    kh, iterations=1)

    contornos, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h < altura * 0.30 or w < 10:
            continue
        candidatos.append((x, y, w, h))

    candidatos.sort(key=lambda b: b[0])

    # Fusionar cajas muy cercanas (mismo dígito partido)
    filtrados = []
    for d in candidatos:
        if filtrados and abs(d[0] - filtrados[-1][0]) < 15:
            if d[3] > filtrados[-1][3]:
                filtrados[-1] = d
        else:
            filtrados.append(d)
    return filtrados


def leer_zona(zone, tipo="temp"):
    if zone is None or zone.size == 0:
        return "--", None, []
    thresh = binarizar(zone)
    bboxes = separar_digitos(thresh)
    if not bboxes:
        return "--", thresh, []

    resultado    = ""
    debug_digitos = []
    for (x, y, w, h) in bboxes:
        recorte = thresh[y:y+h, x:x+w]
        if recorte.shape[0] < 20 or recorte.shape[1] < 8:
            continue
        char, segs, zonas = leer_digito(recorte)
        resultado += char
        debug_digitos.append((segs, zonas, (x, y, w, h), recorte))

    resultado = re.sub(r'[^0-9]', '', resultado)
    if not resultado or len(resultado) > 3:
        return "--", thresh, debug_digitos
    return resultado, thresh, debug_digitos


def leer_decimal(zone):
    if zone is None or zone.size == 0:
        return '?', None, []
    thresh = binarizar(zone)
    bboxes = separar_digitos(thresh)
    if not bboxes:
        return '?', thresh, []
    bboxes = sorted(bboxes, key=lambda b: b[2]*b[3], reverse=True)
    x, y, w, h = bboxes[0]
    recorte = thresh[y:y+h, x:x+w]
    if recorte.shape[0] < 20 or recorte.shape[1] < 8:
        return '?', thresh, []
    char, segs, zonas = leer_digito(recorte)
    return char, thresh, [(segs, zonas, (x, y, w, h), recorte)]


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTADO POR CÁMARA
# ═══════════════════════════════════════════════════════════════════════════════
class EstadoCamara:
    def __init__(self, cfg):
        self.cfg    = cfg
        self.cam    = None
        self.activa = False

        self.votos_temp = deque(maxlen=VOTE_N)
        self.votos_hum  = deque(maxlen=VOTE_N)

        # Última lectura numérica válida para el filtro físico
        self._prev_temp = None
        self._prev_hum  = None

        self.debug = {
            "temp_texto_raw": "", "temp_procesada": None, "temp_dbg_segs": [],
            "hum_texto_raw":  "", "hum_procesada":  None, "hum_dbg_segs":  [],
            "dec_texto_raw":  "", "dec_procesada":  None, "dec_dbg_segs":  [],
            "temp_raw": "--", "hum_raw": "--",
        }
        self.ultima_temp = "--"
        self.ultima_hum  = "--"

    def abrir(self):
        self.cam = cv2.VideoCapture(self.cfg["index"])
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cam.isOpened()

    def _votar(self, cola, valor):
        """
        Votación por mayoría.
        Si el valor actual es válido (no '--') y difiere de todos los anteriores
        en más de 1 unidad, limpia el buffer para no arrastrar lecturas viejas.
        Esto evita que 28.4 en el buffer gane contra el 27.5 correcto actual.
        """
        if valor != '--':
            # Verificar si el nuevo valor es muy distinto al historial reciente
            if cola:
                try:
                    v_nuevo = float(valor)
                    v_prev  = float(list(cola)[-1])
                    # Si difiere >2 del último valor del buffer, limpiar buffer
                    # (cambio real, no ruido)
                    if abs(v_nuevo - v_prev) > 2:
                        cola.clear()
                except (ValueError, TypeError):
                    pass
            cola.append(valor)
        if not cola:
            return '--'
        return Counter(cola).most_common(1)[0][0]

    def _filtro_fisico(self, valor_str, prev, max_delta):
        """Rechaza lecturas que cambian más de max_delta respecto a la anterior."""
        if valor_str == '--' or prev is None:
            return valor_str
        try:
            v = float(valor_str)
            if abs(v - prev) > max_delta:
                return '--'   # salto imposible → descartar
            return valor_str
        except ValueError:
            return '--'

    def leer_temperatura(self, frame):
        tc  = self.cfg["temp_coords"]
        tdc = self.cfg["temp_dec_coords"]
        ze  = frame[tc[1]:tc[3],   tc[0]:tc[2]]
        zd  = frame[tdc[1]:tdc[3], tdc[0]:tdc[2]]

        parte_entera,  thresh_e, dbg_e = leer_zona(ze,  tipo="temp")
        parte_decimal, thresh_d, dbg_d = leer_decimal(zd)

        self.debug["temp_texto_raw"] = parte_entera
        self.debug["temp_procesada"] = thresh_e
        self.debug["temp_dbg_segs"]  = dbg_e
        self.debug["dec_texto_raw"]  = parte_decimal
        self.debug["dec_procesada"]  = thresh_d
        self.debug["dec_dbg_segs"]   = dbg_d

        if parte_entera == "--":
            self.debug["temp_raw"] = "--"
            return self._votar(self.votos_temp, "--")

        texto = f"{parte_entera}.{parte_decimal}" \
                if parte_decimal in '0123456789' else parte_entera
        try:
            valor = float(texto)
            res   = (f"{valor:.1f}" if '.' in texto else f"{valor:.0f}") \
                    if -10 <= valor <= 60 else "--"
        except ValueError:
            res = "--"

        # Filtro físico: descartar saltos imposibles
        res = self._filtro_fisico(res, self._prev_temp, MAX_DELTA_TEMP)
        if res != '--':
            self._prev_temp = float(res)

        self.debug["temp_raw"] = res
        return self._votar(self.votos_temp, res)

    def leer_humedad(self, frame):
        hc   = self.cfg["hum_coords"]
        zone = frame[hc[1]:hc[3], hc[0]:hc[2]]

        raw, thresh, dbg = leer_zona(zone, tipo="hum")
        self.debug["hum_texto_raw"] = raw
        self.debug["hum_procesada"] = thresh
        self.debug["hum_dbg_segs"]  = dbg

        if raw == "--":
            self.debug["hum_raw"] = "--"
            return self._votar(self.votos_hum, "--")
        try:
            valor = float(raw)
            res   = f"{valor:.0f}" if 0 <= valor <= 100 else "--"
        except ValueError:
            res = "--"

        res = self._filtro_fisico(res, self._prev_hum, MAX_DELTA_HUM)
        if res != '--':
            self._prev_hum = float(res)

        self.debug["hum_raw"] = res
        return self._votar(self.votos_hum, res)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ POR CÁMARA
# ═══════════════════════════════════════════════════════════════════════════════
class InterfazCamara:
    def __init__(self, parent_notebook, estado: EstadoCamara, app):
        self.estado      = estado
        self.app         = app
        self.frame_count = 0

        # ── Pestaña ──────────────────────────────────────────────────────────
        self.tab = tk.Frame(parent_notebook, bg="#1e1e1e")
        parent_notebook.add(self.tab, text=estado.cfg["nombre"])

        # ── Video + datos ────────────────────────────────────────────────────
        frame_top = tk.Frame(self.tab, bg="#1e1e1e")
        frame_top.pack(padx=10, pady=(10, 4))

        self.video_label = tk.Label(frame_top, bg="#1e1e1e")
        self.video_label.pack(side=tk.LEFT, padx=(0, 14))

        frame_datos = tk.Frame(frame_top, bg="#1e1e1e")
        frame_datos.pack(side=tk.LEFT, anchor="center")

        tk.Label(frame_datos, text=estado.cfg["nombre"].upper(),
                 font=("Arial", 11, "bold"), fg="#888888", bg="#1e1e1e").pack(pady=(0, 4))
        tk.Label(frame_datos, text="LECTURAS EN VIVO",
                 font=("Arial", 9), fg="#555555", bg="#1e1e1e").pack(pady=(0, 10))

        self.temp_label = tk.Label(frame_datos, text="🌡  -- °C",
                                   font=("Arial", 26, "bold"), fg="#00ff88",
                                   bg="#1e1e1e", width=12, anchor="w")
        self.temp_label.pack(pady=6)

        self.hum_label = tk.Label(frame_datos, text="💧 -- %",
                                  font=("Arial", 26, "bold"), fg="#00bfff",
                                  bg="#1e1e1e", width=12, anchor="w")
        self.hum_label.pack(pady=6)

        self.estado_label = tk.Label(frame_datos, text="⏸  Detenido",
                                     font=("Arial", 11), fg="gray", bg="#1e1e1e")
        self.estado_label.pack(pady=(14, 6))

        tk.Button(frame_datos, text="▶   Comenzar medición",
                  command=self._comenzar, bg="#00aa55", fg="white",
                  font=("Arial", 11, "bold"), width=20, pady=6).pack(pady=4)
        tk.Button(frame_datos, text="⏹   Parar medición",
                  command=self._parar, bg="#cc2200", fg="white",
                  font=("Arial", 11, "bold"), width=20, pady=6).pack(pady=4)

        # ── Historial ────────────────────────────────────────────────────────
        tk.Label(self.tab, text="HISTORIAL",
                 font=("Arial", 10, "bold"), fg="#555555", bg="#1e1e1e").pack()
        self.historial = tk.Listbox(self.tab, width=70, height=7,
                                    bg="#2d2d2d", fg="white", font=("Courier", 10))
        self.historial.pack(padx=10, pady=(2, 8))

        # ── Panel de diagnóstico ─────────────────────────────────────────────
        frame_diag = tk.Frame(self.tab, bg="#111111", relief="solid", bd=1)
        frame_diag.pack(padx=10, pady=(0, 10), fill="x")

        tk.Label(frame_diag,
                 text=f"DIAGNÓSTICO – Segmentos LCD  [{estado.cfg['nombre']}]",
                 font=("Courier", 9, "bold"), fg="#ffaa00", bg="#111111").pack(pady=(6, 2))

        self.diag_canvas = tk.Canvas(frame_diag, width=460, height=340,
                                     bg="#111111", highlightthickness=0)
        self.diag_canvas.pack(padx=8, pady=4)

        self.diag_result = tk.Label(frame_diag, text="Esperando lectura...",
                                    font=("Courier", 12, "bold"),
                                    fg="#00ff88", bg="#111111")
        self.diag_result.pack(pady=(2, 8))

        if not estado.cam or not estado.cam.isOpened():
            self._mostrar_sin_camara()

    # ── Control ───────────────────────────────────────────────────────────────
    def _comenzar(self):
        self.estado.activa = True
        self.estado_label.config(text="🟢 Midiendo...", fg="#00ff88")

    def _parar(self):
        self.estado.activa = False
        self.estado_label.config(text="⏸  Detenido", fg="gray")

    # ── Loop de video (30 ms por frame, OCR cada INTERVALO_MS) ───────────────
    def update(self):
        if not self.estado.cam or not self.estado.cam.isOpened():
            self.tab.after(30, self.update)
            return

        ret, frame = self.estado.cam.read()
        if ret:
            # Dibujar ROIs sobre el frame
            disp = frame.copy()
            tc, tdc, hc = (self.estado.cfg["temp_coords"],
                           self.estado.cfg["temp_dec_coords"],
                           self.estado.cfg["hum_coords"])
            cv2.rectangle(disp, (tc[0],tc[1]),   (tc[2],tc[3]),   (0,255,0),    2)
            cv2.putText(disp, "TEMP", (tc[0],  max(tc[1]-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.rectangle(disp, (tdc[0],tdc[1]), (tdc[2],tdc[3]), (0,220,220),  2)
            cv2.putText(disp, ".DEC", (tdc[0], max(tdc[1]-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,220,220), 2)
            cv2.rectangle(disp, (hc[0],hc[1]),   (hc[2],hc[3]),   (100,180,255),2)
            cv2.putText(disp, "HUM",  (hc[0],  max(hc[1]-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,180,255), 2)

            small = cv2.resize(disp, (DISPLAY_W, DISPLAY_H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # OCR cada INTERVALO_MS milisegundos
            # A 30ms/frame → cada ~67 frames sería 2 seg; usamos tiempo real
            if self.estado.activa:
                ahora = datetime.datetime.now()
                if not hasattr(self, '_ultima_lectura'):
                    self._ultima_lectura = ahora - datetime.timedelta(seconds=10)

                if (ahora - self._ultima_lectura).total_seconds() * 1000 >= INTERVALO_MS:
                    self._ultima_lectura = ahora
                    self._hacer_lectura(frame, ahora)

        self.frame_count += 1
        self.tab.after(30, self.update)

    def _hacer_lectura(self, frame, ahora):
        temp = self.estado.leer_temperatura(frame)
        hum  = self.estado.leer_humedad(frame)

        self.estado.ultima_temp = temp
        self.estado.ultima_hum  = hum

        self.temp_label.config(text=f"🌡  {temp} °C")
        self.hum_label.config(text=f"💧 {hum} %")
        self._actualizar_diagnostico()

        ts = ahora.strftime("%Y-%m-%d %H:%M:%S")
        self.historial.insert(
            tk.END,
            f"{ahora.strftime('%H:%M:%S')}  |  Temp: {temp} °C  |  Hum: {hum} %"
        )
        self.historial.see(tk.END)

        with open(self.estado.cfg["csv_individual"], "a",
                  newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, temp, hum])

        self.app.registrar_lectura_combinada(ahora)

    def _mostrar_sin_camara(self):
        self.video_label.config(
            text=f"❌  {self.estado.cfg['nombre']}\nno disponible\n"
                 f"(index={self.estado.cfg['index']})",
            fg="#ff4444", font=("Arial", 12, "bold"),
            width=30, height=8, relief="solid", bd=1
        )

    # ── Panel de diagnóstico ──────────────────────────────────────────────────
    def _actualizar_diagnostico(self):
        c = self.diag_canvas
        d = self.estado.debug
        c.delete("all")
        if hasattr(c, '_imgs'):
            c._imgs.clear()

        # ─ Imágenes binarizadas con segmentos superpuestos ───────────────────
        c.create_text(8, 8, anchor="nw",
            text="Binarizado + segmentos  (verde=activo  rojo=inactivo):",
            font=("Courier", 8, "bold"), fill="#888888")

        IMG_H, IMG_W = 80, 135
        self._poner_img(c, d["temp_procesada"],  8,  22, IMG_W, IMG_H,
                        "TEMP", "#00ff88", d.get("temp_dbg_segs"))
        self._poner_img(c, d["dec_procesada"],  155, 22,  65,   IMG_H,
                        ".DEC", "#00ddff", d.get("dec_dbg_segs"))
        self._poner_img(c, d["hum_procesada"],  232, 22, IMG_W, IMG_H,
                        "HUM",  "#00bfff", d.get("hum_dbg_segs"))

        # ─ Texto crudo detectado ─────────────────────────────────────────────
        y = 112
        c.create_text(8, y, anchor="nw", text="Texto detectado por OCR:",
                      font=("Courier", 8, "bold"), fill="#888888")
        for i, (zona, clave, col) in enumerate([
            ("TEMP", "temp_texto_raw", "#00ff88"),
            (".DEC", "dec_texto_raw",  "#00ddff"),
            ("HUM",  "hum_texto_raw",  "#00bfff"),
        ]):
            txt = d[clave]
            yy  = y + 18 + i * 20
            msg = f'[{zona}]  "{txt}"' if txt not in ("", "--") else f"[{zona}]  (sin texto)"
            c.create_text(8, yy + 8, anchor="w", text=msg,
                          font=("Courier", 9),
                          fill=col if txt not in ("", "--") else "#555555")

        # ─ Tupla de segmentos (a f b g e c d) ────────────────────────────────
        y2 = y + 82
        c.create_line(0, y2-3, 460, y2-3, fill="#2a2a2a", width=1)
        c.create_text(8, y2, anchor="nw",
                      text="Segmentos activos detectados  (a  f  b  g  e  c  d):",
                      font=("Courier", 8, "bold"), fill="#888888")
        for i, (zona, clave, col) in enumerate([
            ("TEMP", "temp_dbg_segs", "#00ff88"),
            (".DEC", "dec_dbg_segs",  "#00ddff"),
            ("HUM",  "hum_dbg_segs",  "#00bfff"),
        ]):
            dbg = d.get(clave) or []
            segs_str = "  ".join(str(s) for s in dbg[0][0]) if dbg else "—"
            yy = y2 + 16 + i * 18
            c.create_text(8, yy, anchor="nw",
                          text=f"[{zona}]  {segs_str}",
                          font=("Courier", 9), fill=col)

        # ─ Valores validados ─────────────────────────────────────────────────
        y3 = y2 + 76
        c.create_line(0, y3-3, 460, y3-3, fill="#2a2a2a", width=1)
        c.create_text(8, y3, anchor="nw", text="Valores validados + filtro físico:",
                      font=("Courier", 8, "bold"), fill="#888888")
        t_r, h_r = d["temp_raw"], d["hum_raw"]
        c.create_text(8, y3+20, anchor="nw", text=f"Temperatura →  {t_r} °C",
                      font=("Courier", 11, "bold"),
                      fill="#00ff88" if t_r != "--" else "#ff4444")
        c.create_text(8, y3+42, anchor="nw", text=f"Humedad     →  {h_r} %",
                      font=("Courier", 11, "bold"),
                      fill="#00bfff" if h_r != "--" else "#ff4444")

        # ─ Buffer de votación ────────────────────────────────────────────────
        y4 = y3 + 68
        c.create_line(0, y4-3, 460, y4-3, fill="#2a2a2a", width=1)
        c.create_text(8, y4, anchor="nw",
                      text=f"Buffer votación (últimas {VOTE_N} lecturas cada {INTERVALO_MS//1000}s):",
                      font=("Courier", 8, "bold"), fill="#888888")
        bt = "  ".join(list(self.estado.votos_temp)) or "(vacío)"
        bh = "  ".join(list(self.estado.votos_hum))  or "(vacío)"
        c.create_text(8, y4+16, anchor="nw", text=f"T: {bt}",
                      font=("Courier", 9), fill="#666666")
        c.create_text(8, y4+32, anchor="nw", text=f"H: {bh}",
                      font=("Courier", 9), fill="#666666")

        self.diag_result.config(text=f"  Temp: {t_r} °C     Hum: {h_r} %  ")

    def _dibujar_segmentos(self, thresh, dbg_segs):
        """Superpone rectángulos verde/rojo por cada segmento detectado."""
        if thresh is None or thresh.size == 0 or not dbg_segs:
            return thresh
        vis    = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        NOMBRES = ['a', 'f', 'b', 'g', 'e', 'c', 'd']
        for (segs, zonas, (dx, dy, dw, dh), _) in dbg_segs:
            for i, (zx1, zy1, zx2, zy2) in enumerate(zonas):
                ax1 = dx + zx1 // 2
                ay1 = dy + zy1 // 2
                ax2 = dx + zx2 // 2
                ay2 = dy + zy2 // 2
                color = (0, 210, 0) if segs[i] else (0, 0, 200)
                cv2.rectangle(vis, (ax1, ay1), (ax2, ay2), color, 1)
                cv2.putText(vis, NOMBRES[i], (ax1+1, ay1+9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                            (0, 255, 200) if segs[i] else (80, 80, 80), 1)
        return vis

    def _poner_img(self, canvas, img, x, y, max_w, max_h,
                   etiqueta, color, dbg_segs=None):
        if img is None or img.size == 0:
            canvas.create_rectangle(x, y, x+max_w, y+max_h,
                                    outline="#333333", fill="#1a1a1a")
            canvas.create_text(x+max_w//2, y+max_h//2,
                                text="sin imagen", fill="#444444",
                                font=("Courier", 8))
            return

        if dbg_segs:
            vis = self._dibujar_segmentos(img, dbg_segs)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_i, w_i = rgb.shape[:2]
        scale     = min(max_w / w_i, max_h / h_i)
        nw, nh    = max(1, int(w_i*scale)), max(1, int(h_i*scale))
        resized   = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        photo     = ImageTk.PhotoImage(image=Image.fromarray(resized))

        canvas.create_image(x, y, anchor="nw", image=photo)
        canvas.create_rectangle(x, y, x+nw, y+nh, outline=color, width=2)
        canvas.create_text(x+4, y+4, anchor="nw", text=etiqueta,
                           fill=color, font=("Courier", 8, "bold"))

        if not hasattr(canvas, '_imgs'):
            canvas._imgs = []
        canvas._imgs.append(photo)
        if len(canvas._imgs) > 40:
            canvas._imgs = canvas._imgs[-40:]


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class App:
    CSV_INTERVALO_SEG = 60   # cada cuántos segundos escribir al CSV combinado

    def __init__(self):
        _init_csvs()

        self.estados = []
        for cfg in CAMARAS:
            e  = EstadoCamara(cfg)
            ok = e.abrir()
            if not ok:
                print(f"⚠️  {cfg['nombre']} (index={cfg['index']}) no disponible.")
            self.estados.append(e)

        self.root = tk.Tk()
        self.root.title("Monitor 4 Cámaras – Temperatura & Humedad")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(True, True)

        self._construir_resumen()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",     background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d2d", foreground="#aaaaaa",
                        font=("Arial", 10, "bold"), padding=[12, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", "#1e1e1e")],
                  foreground=[("selected", "#00ff88")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.paneles = []
        for estado in self.estados:
            panel = InterfazCamara(self.notebook, estado, self)
            self.paneles.append(panel)

        self._ultimo_csv = datetime.datetime.now()

        for panel in self.paneles:
            panel.update()

        self.root.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.root.mainloop()

    def _construir_resumen(self):
        frame_res = tk.Frame(self.root, bg="#111111")
        frame_res.pack(fill="x")
        tk.Label(frame_res, text="RESUMEN – TODAS LAS CÁMARAS",
                 font=("Arial", 9, "bold"), fg="#555555", bg="#111111"
                 ).pack(side=tk.LEFT, padx=12)
        self.resumen_labels = []
        for cfg in CAMARAS:
            lbl = tk.Label(frame_res,
                           text=f"{cfg['nombre']}: -- °C | -- %",
                           font=("Arial", 10, "bold"), fg="#888888", bg="#111111")
            lbl.pack(side=tk.LEFT, padx=14)
            self.resumen_labels.append(lbl)

    def registrar_lectura_combinada(self, ahora):
        for i, estado in enumerate(self.estados):
            self.resumen_labels[i].config(
                text=f"{estado.cfg['nombre']}: {estado.ultima_temp} °C | {estado.ultima_hum} %"
            )
        if (ahora - self._ultimo_csv).total_seconds() >= self.CSV_INTERVALO_SEG:
            ts   = ahora.strftime("%Y-%m-%d %H:%M:%S")
            fila = [ts]
            for estado in self.estados:
                fila += [estado.ultima_temp, estado.ultima_hum]
            with open(CSV_COMBINADO, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fila)
            self._ultimo_csv = ahora

    def _cerrar(self):
        for estado in self.estados:
            if estado.cam:
                estado.cam.release()
        self.root.destroy()


if __name__ == "__main__":
    App()