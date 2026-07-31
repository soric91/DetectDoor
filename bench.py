"""Mide latencia y RAM del detector en el equipo donde va a correr.

Uso en el IOT-GATE-iMX8:
    uv run python bench.py                       # usa el stream RTSP
    uv run python bench.py video.mp4             # usa un archivo

Con la latencia media que imprime (L), los parametros del .env salen asi:
    ANALYSIS_INTERVAL -> 0 si L >= 1s (la inferencia ya es el cuello de botella)
    DWELL_SECONDS     -> >= 3 * L, para exigir varias muestras antes de avisar
    TRACK_TTL         -> >= 2 * (ANALYSIS_INTERVAL + L), si no la misma persona
                         recibe un id nuevo y vuelve a alertar
"""

import resource
import statistics
import sys
import time

import cv2

import main


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main_bench():
    fuente = sys.argv[1] if len(sys.argv) > 1 else main.RTSP_URL
    frames_objetivo = 30

    print(f"[bench] RSS base                : {rss_mb():.0f} MB")
    detector = main.Detector()
    print(f"[bench] RSS con modelo cargado  : {rss_mb():.0f} MB")
    print(f"[bench] imgsz={main.IMGSZ} hilos={main.NCNN_THREADS}")

    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"[bench] no se pudo abrir {fuente}")
        return 1

    cx1, cy1, cx2, cy2 = main.recorte_roi()
    print(f"[bench] recorte: {cx2 - cx1}x{cy2 - cy1} px (frame completo: {main.ANCHO}x{main.ALTO})")

    gate = main.GateMovimiento()
    latencias_crop, latencias_full, con_movimiento = [], [], 0

    for i in range(frames_objetivo):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != main.ANCHO or frame.shape[0] != main.ALTO:
            frame = cv2.resize(frame, (main.ANCHO, main.ALTO))
        recorte = frame[cy1:cy2, cx1:cx2]

        if gate.hay_movimiento(recorte):
            con_movimiento += 1

        t = time.monotonic()
        detector.detect(recorte)
        latencias_crop.append(time.monotonic() - t)

        # Comparacion contra analizar el frame entero, solo en unos pocos frames
        if i < 5:
            t = time.monotonic()
            detector.detect(frame)
            latencias_full.append(time.monotonic() - t)

    cap.release()
    if not latencias_crop:
        print("[bench] no se leyo ningun frame")
        return 1

    # Descarta el warmup
    crop = latencias_crop[2:] or latencias_crop
    full = latencias_full[2:] or latencias_full
    l_crop = statistics.mean(crop)

    print(f"[bench] RSS pico                : {rss_mb():.0f} MB")
    print(f"[bench] latencia recorte        : {l_crop * 1000:.0f} ms  ({1 / l_crop:.1f} FPS)")
    if full:
        print(f"[bench] latencia frame completo : {statistics.mean(full) * 1000:.0f} ms")
    print(f"[bench] frames con movimiento   : {con_movimiento}/{len(latencias_crop)}")
    print()
    print("Parametros sugeridos para el .env:")
    print(f"  ANALYSIS_INTERVAL={0 if l_crop >= 1 else 1}")
    print(f"  DWELL_SECONDS={max(3, round(3 * l_crop))}")
    intervalo = 0 if l_crop >= 1 else 1
    print(f"  TRACK_TTL={max(3, round(2 * (intervalo + l_crop)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main_bench())
