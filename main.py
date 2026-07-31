"""Detector de personas en la puerta para IOT-GATE-iMX8 (Cortex-A53, 945 MB, sin NPU).

Corre YOLO11n sobre NCNN directamente, sin ultralytics ni torch: el import de
ultralytics arrastra torch, que solo no entra en la RAM de este equipo. Con NCNN
puro el proceso entra comodo y nunca toca swap.

Tres optimizaciones para el A53:
  1. Gate de movimiento: solo se analiza cuando algo cambia en el recuadro.
  2. Crop del recuadro: se le pasa al modelo la puerta, no los 640x360 completos.
  3. Tracker IoU propio, en vez del ByteTrack de ultralytics.
"""

import logging
import os
import sys
import time
import threading
import signal

import cv2
import numpy as np
import ncnn
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("viewcam")


def configurar_log():
    """Log a stdout. Bajo systemd se omite el timestamp: journalctl ya pone uno."""
    bajo_systemd = "JOURNAL_STREAM" in os.environ
    formato = "%(levelname)s %(name)s: %(message)s"
    if not bajo_systemd:
        formato = "%(asctime)s " + formato

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format=formato,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def env_str(name, default=None):
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Falta la variable {name} en el .env")
    return value


def env_int(name, default):
    return int(os.getenv(name, default))


def env_float(name, default):
    return float(os.getenv(name, default))


def env_bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "si", "sí", "on")


# --- Configuracion ---
RTSP_URL = env_str("RTSP_URL", "rtsp://localhost:8554/frente")
# Pipeline GStreamer opcional, para decodificar H264 con la VPU del i.MX8.
# Si esta vacio se usa el backend por defecto (decodifica en CPU).
CAPTURE_PIPELINE = os.getenv("CAPTURE_PIPELINE", "").strip()

TOKEN_TELEGRAM = env_str("TOKEN_TELEGRAM")
CHAT_ID = env_str("CHAT_ID")

ANCHO = env_int("FRAME_WIDTH", 640)
ALTO = env_int("FRAME_HEIGHT", 360)

# Recuadro de la puerta (zona de deteccion)
ROI_X = env_int("ROI_X", 440)
ROI_Y = env_int("ROI_Y", 80)
ROI_W = env_int("ROI_W", 120)
ROI_H = env_int("ROI_H", 230)
# Margen extra alrededor del recuadro al recortar, para no cortar el cuerpo
ROI_PAD = env_float("ROI_PAD", 0.25)

MODEL_DIR = env_str("NCNN_MODEL_DIR", "yolo11n_ncnn_model")
IMGSZ = env_int("IMGSZ", 256)
NCNN_THREADS = env_int("NCNN_THREADS", 4)
CONF = env_float("YOLO_CONF", 0.4)
NMS_IOU = env_float("NMS_IOU", 0.45)

# Segundos que la persona debe permanecer en el recuadro antes de avisar
DWELL_SECONDS = env_float("DWELL_SECONDS", 3.0)
# Segundos entre analisis. 0 = tan rapido como pueda el equipo.
ANALYSIS_INTERVAL = env_float("ANALYSIS_INTERVAL", 0.0)
# Segundos sin ver un track antes de olvidarlo. Debe ser > tiempo entre analisis.
TRACK_TTL = env_float("TRACK_TTL", 3.0)
# IoU minimo para considerar que dos cajas son la misma persona
IOU_MATCH = env_float("IOU_MATCH", 0.3)
# Fraccion del cuerpo que debe caer dentro del recuadro
OVERLAP_MIN = env_float("OVERLAP_MIN", 0.35)

# Gate de movimiento
MOTION_ENABLED = env_bool("MOTION_ENABLED", "true")
# Fraccion de pixeles del recuadro que deben cambiar para despertar el detector
MOTION_AREA = env_float("MOTION_AREA", 0.01)
# Cuanta diferencia de intensidad cuenta como cambio (0-255)
MOTION_DELTA = env_int("MOTION_DELTA", 25)
# Segundos que se sigue analizando despues del ultimo movimiento
MOTION_HOLD = env_float("MOTION_HOLD", 3.0)

SHOW_VIDEO = env_bool("SHOW_VIDEO", "false")
RECONNECT_SECONDS = env_float("RECONNECT_SECONDS", 5.0)
MENSAJE = env_str("MENSAJE_ALERTA", "🚪 Hay alguien en la puerta")


# --- Telegram ---
def enviar_alerta(frame):
    """Envia el frame limpio (sin recuadro) a Telegram en segundo plano."""
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        log.error("no se pudo codificar el frame para enviar")
        return

    def worker(jpg_bytes):
        try:
            envio = time.monotonic()
            response = requests.post(
                f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendPhoto",
                data={"chat_id": CHAT_ID, "caption": MENSAJE},
                files={"photo": ("puerta.jpg", jpg_bytes, "image/jpeg")},
                timeout=15,
            )
            if response.status_code == 200:
                log.info("alerta enviada en %.1fs", time.monotonic() - envio)
            else:
                log.error("telegram respondio %s: %s", response.status_code, response.text)
        except requests.RequestException as exc:
            log.error("fallo el envio a telegram: %s", exc)

    threading.Thread(target=worker, args=(buffer.tobytes(),), daemon=True).start()


# --- Detector NCNN ---
class Detector:
    """YOLO11n sobre NCNN. Devuelve solo personas (clase 0)."""

    PERSON_CLASS = 0

    def __init__(self):
        self.net = ncnn.Net()
        self.net.opt.num_threads = NCNN_THREADS
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(os.path.join(MODEL_DIR, "model.ncnn.param"))
        self.net.load_model(os.path.join(MODEL_DIR, "model.ncnn.bin"))

    def _letterbox(self, img):
        """Escala manteniendo proporcion y rellena hasta IMGSZ. Devuelve (img, escala, dx, dy)."""
        h, w = img.shape[:2]
        escala = min(IMGSZ / w, IMGSZ / h)
        nw, nh = max(1, round(w * escala)), max(1, round(h * escala))
        redim = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lienzo = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
        dx, dy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
        lienzo[dy:dy + nh, dx:dx + nw] = redim
        return lienzo, escala, dx, dy

    def detect(self, img):
        """Detecta personas. Devuelve [(x1, y1, x2, y2, score)] en coordenadas de img."""
        lienzo, escala, dx, dy = self._letterbox(img)

        mat = ncnn.Mat.from_pixels(lienzo, ncnn.Mat.PixelType.PIXEL_BGR2RGB, IMGSZ, IMGSZ)
        mat.substract_mean_normalize([], [1 / 255.0] * 3)

        with self.net.create_extractor() as ex:
            ex.input("in0", mat)
            _, out = ex.extract("out0")
            salida = np.array(out)  # (84, N): 4 bbox + 80 clases

        scores = salida[4 + self.PERSON_CLASS]
        keep = scores >= CONF
        if not keep.any():
            return []

        cx, cy, bw, bh = salida[0][keep], salida[1][keep], salida[2][keep], salida[3][keep]
        scores = scores[keep]

        # De centro/tamano en el lienzo a esquinas en coordenadas de img
        x1 = (cx - bw / 2 - dx) / escala
        y1 = (cy - bh / 2 - dy) / escala
        x2 = (cx + bw / 2 - dx) / escala
        y2 = (cy + bh / 2 - dy) / escala

        cajas = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(cajas.tolist(), scores.tolist(), CONF, NMS_IOU)
        if len(indices) == 0:
            return []

        h, w = img.shape[:2]
        salida_final = []
        for i in np.array(indices).flatten():
            salida_final.append((
                float(np.clip(x1[i], 0, w)),
                float(np.clip(y1[i], 0, h)),
                float(np.clip(x2[i], 0, w)),
                float(np.clip(y2[i], 0, h)),
                float(scores[i]),
            ))
        return salida_final


# --- Tracker IoU ---
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class Tracker:
    """Asocia cajas entre analisis por IoU. Suficiente para 1-2 personas en una puerta."""

    def __init__(self):
        self.tracks = {}  # id -> {"caja": (x1,y1,x2,y2), "visto": t}
        self._siguiente_id = 1

    def update(self, detecciones, ahora):
        """Devuelve [(x1, y1, x2, y2, track_id)]."""
        for tid, t in list(self.tracks.items()):
            if ahora - t["visto"] > TRACK_TTL:
                del self.tracks[tid]

        libres = set(self.tracks)
        resultado = []

        # Emparejado greedy: cada deteccion toma el track libre con mas solape
        for caja in detecciones:
            mejor_tid, mejor_iou = None, IOU_MATCH
            for tid in libres:
                valor = iou(caja, self.tracks[tid]["caja"])
                if valor >= mejor_iou:
                    mejor_tid, mejor_iou = tid, valor

            if mejor_tid is None:
                mejor_tid = self._siguiente_id
                self._siguiente_id += 1
            else:
                libres.discard(mejor_tid)

            self.tracks[mejor_tid] = {"caja": caja[:4], "visto": ahora}
            resultado.append((*caja[:4], mejor_tid))

        return resultado

    def reset(self):
        self.tracks.clear()


# --- Zona de la puerta ---
def recorte_roi():
    """Region a recortar: el recuadro mas un margen, recortado al frame."""
    px, py = int(ROI_W * ROI_PAD), int(ROI_H * ROI_PAD)
    x1 = max(0, ROI_X - px)
    y1 = max(0, ROI_Y - py)
    x2 = min(ANCHO, ROI_X + ROI_W + px)
    y2 = min(ALTO, ROI_Y + ROI_H + py)
    return x1, y1, x2, y2


def en_recuadro(x1, y1, x2, y2):
    """True si la persona esta dentro del recuadro de la puerta.

    Usa los pies (centro del borde inferior) o un solape minimo del cuerpo,
    asi una persona parada en la puerta cuenta aunque la caja la sobrepase.
    """
    pie_x = (x1 + x2) / 2
    pie_y = y2
    if ROI_X <= pie_x <= ROI_X + ROI_W and ROI_Y <= pie_y <= ROI_Y + ROI_H:
        return True

    inter_w = max(0, min(x2, ROI_X + ROI_W) - max(x1, ROI_X))
    inter_h = max(0, min(y2, ROI_Y + ROI_H) - max(y1, ROI_Y))
    area_persona = max(1, (x2 - x1) * (y2 - y1))
    return (inter_w * inter_h) / area_persona >= OVERLAP_MIN


def evaluar_dwell(seguidas, personas, tracks_vivos, ahora):
    """Decide a quien hay que avisar.

    Devuelve (ids_a_avisar, cajas) donde cajas lleva el flag de dentro/fuera.
    Avisa una sola vez por track_id: mientras sea la misma persona no repite.
    Cuando el track muere, se olvida su estado, asi el proximo que llegue tendra
    un id nuevo y volvera a avisar.
    """
    avisar = []
    cajas = []

    for x1, y1, x2, y2, track_id in seguidas:
        dentro = en_recuadro(x1, y1, x2, y2)
        cajas.append((x1, y1, x2, y2, track_id, dentro))
        if not dentro:
            continue

        estado = personas.get(track_id)
        if estado is None:
            estado = {"desde": ahora, "avisado": False}
            personas[track_id] = estado

        permanencia = ahora - estado["desde"]
        if not estado["avisado"] and permanencia >= DWELL_SECONDS:
            estado["avisado"] = True
            avisar.append((track_id, permanencia))

    for track_id in [t for t in personas if t not in tracks_vivos]:
        del personas[track_id]

    return avisar, cajas


class GateMovimiento:
    """Compara el recorte contra el anterior. Barato: evita correr YOLO sin motivo."""

    def __init__(self):
        self.previo = None

    def hay_movimiento(self, recorte):
        gris = cv2.GaussianBlur(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.previo is None or self.previo.shape != gris.shape:
            self.previo = gris
            return True  # primer frame: analizar por si ya hay alguien

        diff = cv2.absdiff(gris, self.previo)
        self.previo = gris
        cambiados = np.count_nonzero(diff >= MOTION_DELTA)
        return cambiados >= MOTION_AREA * gris.size

    def reset(self):
        self.previo = None


# --- Video ---
def abrir_stream():
    if CAPTURE_PIPELINE:
        cap = cv2.VideoCapture(CAPTURE_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def mostrar(frame, cajas):
    """Dibuja cajas y recuadro sobre la ventana. False si se pidio salir."""
    for x1, y1, x2, y2, track_id, dentro in cajas:
        color = (0, 0, 255) if dentro else (200, 200, 200)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"id {track_id}", (int(x1), max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
    cv2.imshow("Frente - WyzeCam V4", frame)
    return (cv2.waitKey(1) & 0xFF) != 27


def main():
    configurar_log()
    corriendo = {"activo": True}

    def parar(signum, frame):
        log.info("recibida senal %s, cerrando", signal.Signals(signum).name)
        corriendo["activo"] = False

    signal.signal(signal.SIGINT, parar)
    signal.signal(signal.SIGTERM, parar)

    log.info("cargando %s (NCNN, imgsz=%d, %d hilos)", MODEL_DIR, IMGSZ, NCNN_THREADS)
    detector = Detector()
    tracker = Tracker()
    gate = GateMovimiento()
    cx1, cy1, cx2, cy2 = recorte_roi()
    log.info("recorte de analisis: %dx%d px en (%d,%d)", cx2 - cx1, cy2 - cy1, cx1, cy1)
    log.info("dwell=%.1fs intervalo=%.1fs ttl=%.1fs gate_movimiento=%s",
             DWELL_SECONDS, ANALYSIS_INTERVAL, TRACK_TTL, MOTION_ENABLED)

    # track_id -> {"desde": primer instante dentro del recuadro,
    #              "avisado": ya se envio la alerta}
    personas = {}
    cap = None
    ultimo_analisis = 0.0
    ultimo_movimiento = 0.0
    ultimas_cajas = []

    while corriendo["activo"]:
        if cap is None:
            cap = abrir_stream()
            if cap is None:
                log.warning("sin conexion al stream, reintento en %.0fs", RECONNECT_SECONDS)
                time.sleep(RECONNECT_SECONDS)
                continue
            log.info("stream conectado")

        ret, frame = cap.read()
        if not ret:
            log.warning("stream perdido, reconectando")
            cap.release()
            cap = None
            tracker.reset()
            gate.reset()
            personas.clear()
            ultimas_cajas = []
            time.sleep(RECONNECT_SECONDS)
            continue

        if frame.shape[1] != ANCHO or frame.shape[0] != ALTO:
            frame = cv2.resize(frame, (ANCHO, ALTO))
        frame_limpio = frame.copy()  # sin dibujos, es el que se envia
        ahora = time.monotonic()

        if ANALYSIS_INTERVAL and ahora - ultimo_analisis < ANALYSIS_INTERVAL:
            if SHOW_VIDEO and not mostrar(frame, ultimas_cajas):
                break
            continue

        recorte = frame[cy1:cy2, cx1:cx2]

        # Gate: se analiza si hay movimiento, si alguien sigue en la puerta
        # (una persona quieta no genera movimiento pero debe seguir contando),
        # o durante unos segundos despues del ultimo movimiento.
        if MOTION_ENABLED:
            if gate.hay_movimiento(recorte):
                ultimo_movimiento = ahora
            analizar = (
                personas
                or tracker.tracks
                or ahora - ultimo_movimiento < MOTION_HOLD
            )
        else:
            analizar = True

        if not analizar:
            if SHOW_VIDEO and not mostrar(frame, []):
                break
            continue

        ultimo_analisis = ahora

        # Detectar sobre el recorte y trasladar las coordenadas al frame completo
        crudas = detector.detect(recorte)
        detecciones = [
            (x1 + cx1, y1 + cy1, x2 + cx1, y2 + cy1) for x1, y1, x2, y2, _ in crudas
        ]
        seguidas = tracker.update(detecciones, ahora)
        avisar, ultimas_cajas = evaluar_dwell(seguidas, personas, tracker.tracks, ahora)

        # Con LOG_LEVEL=DEBUG se ve la latencia real de cada analisis, util para
        # ajustar ANALYSIS_INTERVAL, DWELL_SECONDS y TRACK_TTL.
        log.debug("analisis en %.0f ms, %d persona(s), ids=%s",
                  (time.monotonic() - ahora) * 1000, len(seguidas),
                  [t[4] for t in seguidas])

        for track_id, permanencia in avisar:
            log.info("persona %d llevaba %.1fs en la puerta, avisando", track_id, permanencia)
            enviar_alerta(frame_limpio)

        if SHOW_VIDEO and not mostrar(frame, ultimas_cajas):
            break

    if cap is not None:
        cap.release()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    log.info("detenido")


if __name__ == "__main__":
    main()
