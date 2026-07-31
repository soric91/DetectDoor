# viewcam — detector de personas en la puerta con alerta por Telegram

Vigila el stream RTSP de una cámara (Wyze Cam V4), detecta personas dentro de un
recuadro definido sobre la puerta y manda un mensaje a Telegram con la captura
del momento cuando alguien se queda ahí más de N segundos.

Pensado para correr 24/7 en un equipo embebido sin GPU: **CompuLab
IOT-GATE-iMX8** (i.MX8M Mini, Cortex-A53 4 núcleos, 945 MB de RAM, sin NPU).

## Qué hace

- Detecta **solo personas** (YOLO11n, clase `person`).
- Solo cuenta lo que pasa **dentro del recuadro** de la puerta, no en la calle.
- Avisa cuando alguien **permanece** más de `DWELL_SECONDS` (evita falsos avisos
  de quien pasa caminando).
- **Un aviso por persona.** Si sigue ahí no repite el mensaje; cuando llega
  alguien distinto, vuelve a avisar. La idea es un aviso por visita.
- La foto que envía es el **frame limpio**, sin el recuadro ni las cajas dibujadas.

## Cómo funciona

```
RTSP ──► frame 640x360 ──► gate de movimiento ──► crop del recuadro
                                  │ (sin cambios: no analiza)
                                  ▼
                          YOLO11n / NCNN ──► tracker IoU ──► dwell ──► Telegram
```

1. **Gate de movimiento**: compara el recuadro contra el frame anterior
   (`absdiff`). Si nada cambió, no corre el detector. En una puerta, eso es el
   99 % del tiempo. Es lo que hace viable el A53.
   Sigue analizando si hay un track vivo o si el último movimiento fue hace menos
   de `MOTION_HOLD` segundos — una persona parada quieta no genera diff, y sin
   esto se perdería la cuenta del dwell.
2. **Crop del recuadro** (más `ROI_PAD` de margen): mejora la precisión porque la
   persona ocupa más píxeles del input, y descarta a quien pasa por la calle.
   No acelera la inferencia: el modelo NCNN tiene input de tamaño fijo.
3. **Detección** con YOLO11n sobre NCNN, sin PyTorch.
4. **Tracker IoU** propio: asigna un id a cada persona entre análisis.
5. **Dwell**: se avisa una sola vez por id. Cuando el track muere
   (`TRACK_TTL` sin verlo), se olvida su estado; el próximo que llegue tendrá un
   id nuevo y volverá a avisar.

## Por qué NCNN y no ultralytics

`ultralytics` importa PyTorch aunque el modelo sea NCNN, y PyTorch solo, apenas
importado, ocupa más de la mitad de la RAM del equipo. Sumado al pico de la
inferencia no entra en 945 MB: el proceso muere por OOM, o sobrevive a costa de
swap con la latencia multiplicada y desgaste de la eMMC.

Por eso el pre/postproceso, el NMS y el tracker están escritos a mano sobre el
runtime `ncnn`, y `ultralytics` quedó como dependencia **solo de exportación**.
Sin él el proceso entra cómodo en RAM y nunca toca swap.

El postproceso propio se validó contra ultralytics sobre el mismo frame: misma
caja con IoU 1.000 y diferencia de confianza 0.000.

De los tres tamaños exportados, `192` es el que viene configurado. `160` es más
rápido pero la confianza sobre la misma persona cae cerca del umbral de 0.4, poco
margen para condiciones de poca luz; `256` es el más preciso y el más lento.
Para elegir con números de tu equipo, usá `bench.py`.

## Requisitos

- Python 3.10+ y [uv](https://docs.astral.sh/uv/).
- Un stream RTSP accesible (en este proyecto la cámara se republica en
  `rtsp://localhost:8554/frente`).
- Un bot de Telegram y el id del chat de destino.

El runtime son 13 paquetes, ninguno de PyTorch ni CUDA: `ncnn`, `numpy`,
`opencv-python`, `python-dotenv`, `requests` y sus transitivos.

## Instalación

En el equipo donde va a correr:

```bash
git clone <este-repo>
cd viewcam
uv sync                  # runtime, sin torch
cp .env.example .env     # completar TOKEN_TELEGRAM y CHAT_ID
```

En la máquina de desarrollo, si además querés re-exportar el modelo:

```bash
uv sync --group export    # agrega ultralytics + torch
```

**No corras `uv sync --group export` en el equipo embebido**: baja PyTorch, que
pesa cientos de megas y no vas a poder usar.

## Configuración

Todo se ajusta por `.env`, sin tocar código. Ver `.env.example` para la lista
completa con comentarios. Las que más importan:

| Variable | Qué hace |
|---|---|
| `TOKEN_TELEGRAM`, `CHAT_ID` | Credenciales del bot. Obligatorias. |
| `ROI_X`, `ROI_Y`, `ROI_W`, `ROI_H` | Recuadro de la puerta, en píxeles. |
| `DWELL_SECONDS` | Segundos de permanencia antes de avisar. |
| `ANALYSIS_INTERVAL` | Segundos entre análisis. `0` = tan rápido como pueda. |
| `TRACK_TTL` | Segundos sin ver a alguien antes de olvidarlo. |
| `IMGSZ` + `NCNN_MODEL_DIR` | Tamaño del modelo. Tienen que coincidir. |
| `MOTION_ENABLED` | Gate de movimiento. Dejar en `true` en equipos lentos. |
| `SHOW_VIDEO` | Ventana de video. `false` en equipos sin escritorio. |

### Calibrar el recuadro

En una máquina con escritorio, poné `SHOW_VIDEO=true` y ajustá
`ROI_X/Y/W/H` hasta que el rectángulo verde cubra la puerta. La ventana muestra
las cajas en rojo cuando la persona cuenta como "en la puerta" y en gris cuando
no, con el id del track. `ESC` cierra.

### Elegir `DWELL_SECONDS` y `TRACK_TTL`

Dependen de la latencia real de tu equipo (`L`). Medila:

```bash
uv run python bench.py                  # sobre el stream RTSP
uv run python bench.py grabacion.mp4    # sobre un archivo
```

Imprime latencia, RSS y los valores sugeridos. La regla:

- `ANALYSIS_INTERVAL` → `0` si `L >= 1s`; la inferencia ya es el cuello de botella
  y saltear frames no ahorra nada.
- `DWELL_SECONDS` → al menos `3 × L`, para exigir varias muestras antes de avisar.
  Si `L` es alto, "3 segundos" no es medible: la granularidad mínima es `L`.
- `TRACK_TTL` → al menos `2 × (ANALYSIS_INTERVAL + L)`. **Si queda por debajo del
  gap real entre análisis, el track muere entre frames, la misma persona recibe un
  id nuevo y vuelve a alertar** — justo el spam que el diseño evita. Si en el log
  ves ids nuevos para alguien que no se movió, subilo.

## Ejecutar

```bash
uv run python main.py
```

Salida normal:

```
[init] cargando yolo11n_ncnn_192 (NCNN, imgsz=192, 4 hilos)
[init] recorte de analisis: 180x337 px en (410,23)
[stream] conectado
[alerta] persona 1 llevaba 3.1s en la puerta
```

Reconecta solo si se cae el stream, y sale limpio con `SIGINT`/`SIGTERM`.

### Como servicio

`/etc/systemd/system/viewcam.service`:

```ini
[Unit]
Description=Detector de personas en la puerta
After=network-online.target

[Service]
Type=simple
User=compulab
WorkingDirectory=/home/compulab/viewcam
ExecStart=/home/compulab/.local/bin/uv run python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now viewcam
journalctl -u viewcam -f
```

## Tests

Prueban la lógica de alertas con tiempo simulado, sin cámara ni modelo:

```bash
uv run python test_alertas.py
```

Cubre: una persona que se queda avisa una sola vez, quien pasa rápido no dispara,
una persona nueva vuelve a avisar, alguien fuera del recuadro nunca avisa, dos
personas simultáneas dan dos avisos, y un parpadeo de detección no duplica el aviso.

## Re-exportar el modelo

Los modelos ya vienen exportados (`yolo11n_ncnn_160/192/256`), así que en el
equipo no necesitás PyTorch. Para regenerarlos, en la máquina de desarrollo:

```bash
uv run --group export yolo export model=yolo11n.pt format=ncnn imgsz=192 half=True
mv yolo11n_ncnn_model yolo11n_ncnn_192
```

Si cambiás `imgsz`, actualizá **las dos** variables (`IMGSZ` y `NCNN_MODEL_DIR`):
el modelo tiene el tamaño de input fijo y no se ajusta solo.

## Problemas comunes

**`uv sync` falla compilando opencv-python en aarch64** — usá el paquete del
sistema: `sudo apt install python3-opencv` y creá el entorno con
`uv venv --system-site-packages`.

**Alertas repetidas de la misma persona** — `TRACK_TTL` está por debajo del gap
real entre análisis. Ver la sección de parámetros.

**Nunca avisa** — revisá con `SHOW_VIDEO=true` que el recuadro cubra la puerta.
Si la persona aparece con caja gris, está fuera del recuadro: ajustá el ROI o bajá
`OVERLAP_MIN`. Si no aparece ninguna caja, bajá `YOLO_CONF` o probá
`NCNN_MODEL_DIR=yolo11n_ncnn_256` con `IMGSZ=256`.

**CPU al 100 % constante** — `MOTION_ENABLED=false` o `ANALYSIS_INTERVAL=0`.
Activá el gate y subí el intervalo.

**El equipo empieza a usar swap** — algo está cargando PyTorch. Verificá que no
se haya instalado `ultralytics`: `uv pip list | grep -i torch` no debe devolver nada.

## Estructura

```
main.py             pipeline completo: captura, gate, detector, tracker, Telegram
bench.py            mide latencia y RAM en el equipo de destino
test_alertas.py     tests de la logica de alertas
.env.example        plantilla de configuracion
yolo11n_ncnn_*/     modelos NCNN exportados (160, 192, 256)
telmsn.py           prueba suelta de la API de Telegram
```

Las grabaciones (`recordings_frente/`) y el `.env` no se suben.

## Notas

- El modelo YOLO11n es de Ultralytics, con licencia **AGPL-3.0**. Tenelo en cuenta
  si vas a distribuir esto.
- El i.MX8M Mini no tiene NPU. Si migrás a un i.MX8M **Plus** (que sí la tiene,
  con `/dev/galcore` y `libvx_delegate.so`), conviene reemplazar NCNN por TFLite
  int8 con el delegate VX, que aprovecha la NPU en vez de la CPU.
