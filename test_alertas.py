"""Prueba la logica de alertas con tiempo simulado, sin camara ni modelo.

    uv run python test_alertas.py
"""

import main

# Caja dentro del recuadro de la puerta (pies dentro del ROI por defecto)
PUERTA_A = (480.0, 120.0, 540.0, 290.0)
PUERTA_B = (460.0, 115.0, 520.0, 295.0)  # otra persona, sin solape con A
CALLE = (20.0, 20.0, 70.0, 180.0)  # fuera del recuadro


def correr(pasos):
    """pasos = [(t, [cajas])]. Devuelve la lista de (t, track_id) que alertaron."""
    tracker = main.Tracker()
    personas = {}
    alertas = []
    for t, cajas in pasos:
        seguidas = tracker.update(list(cajas), t)
        avisar, _ = main.evaluar_dwell(seguidas, personas, tracker.tracks, t)
        for track_id, _permanencia in avisar:
            alertas.append((t, track_id))
    return alertas


def check(nombre, obtenido, esperado):
    ok = obtenido == esperado
    print(f"{'OK  ' if ok else 'FALLA'} {nombre}")
    if not ok:
        print(f"      esperado={esperado}  obtenido={obtenido}")
    return ok


def main_test():
    main.DWELL_SECONDS = 3.0
    main.TRACK_TTL = 5.0
    resultados = []

    # Una persona que se queda: avisa una vez a los 3s y no repite
    alertas = correr([(t, [PUERTA_A]) for t in (0, 1, 2, 3, 4, 5, 6, 7, 8)])
    resultados.append(check("persona que se queda avisa una sola vez",
                            [(t, i) for t, i in alertas], [(3.0, 1)]))

    # Alguien que pasa rapido (menos de DWELL) no genera alerta
    alertas = correr([(0, [PUERTA_A]), (1, [PUERTA_A]), (2, [])])
    resultados.append(check("paso rapido no alerta", alertas, []))

    # Persona A se va, llega B: alerta nueva. La ventana sin nadie debe superar
    # TRACK_TTL para que el track de A muera.
    pasos = [(t, [PUERTA_A]) for t in (0, 1, 2, 3, 4)]
    pasos += [(t, []) for t in (5, 7, 9, 11, 13)]  # nadie, A expira
    pasos += [(t, [PUERTA_B]) for t in (14, 15, 16, 17, 18)]
    alertas = correr(pasos)
    resultados.append(check("persona nueva vuelve a alertar",
                            [tid for _, tid in alertas], [1, 2]))

    # Alguien en la calle, fuera del recuadro, nunca alerta
    alertas = correr([(t, [CALLE]) for t in range(0, 10)])
    resultados.append(check("fuera del recuadro no alerta", alertas, []))

    # Dos personas a la vez: una alerta por cada una
    pasos = [(t, [PUERTA_A, PUERTA_B]) for t in (0, 1, 2, 3, 4)]
    alertas = correr(pasos)
    resultados.append(check("dos personas a la vez dan dos alertas",
                            sorted(tid for _, tid in alertas), [1, 2]))

    # Parpadeo de deteccion menor a TRACK_TTL: sigue siendo la misma persona,
    # no debe re-alertar ni reiniciar la cuenta.
    pasos = [(0, [PUERTA_A]), (1, [PUERTA_A]), (2, []), (3, [PUERTA_A]),
             (4, [PUERTA_A]), (5, [PUERTA_A]), (6, [PUERTA_A])]
    alertas = correr(pasos)
    resultados.append(check("parpadeo no duplica alerta",
                            [tid for _, tid in alertas], [1]))

    print()
    if all(resultados):
        print(f"todo bien ({len(resultados)}/{len(resultados)})")
        return 0
    print(f"fallaron {resultados.count(False)}/{len(resultados)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main_test())
