"""
generar_matriz_confusion_estructura.py
Genera una figura didactica que muestra la ESTRUCTURA de la matriz de
confusion binaria TAL COMO SE USA EN EL PROYECTO (misma convencion que
Datasets/graficos_metricas.py y nucleo.py):

    fake = 0 (clase positiva), real = 1 (clase negativa)
    fila 0 = verdadero FAKE   | fila 1 = verdadero REAL
    columna 0 = predicho FAKE | columna 1 = predicho REAL

    VP = fake -> fake (m[0,0])   FN = fake -> real (m[0,1])
    FP = real -> fake (m[1,0])   VN = real -> real (m[1,1])

No contiene datos numericos reales de ningun experimento (es un esquema),
pero respeta el orden de clases y la definicion de VP/FN/FP/VN que usan
todos los resultados del proyecto (precision_fake, recall_fake, f1_fake).

Salida: figuras/diagramas/matriz_confusion_estructura.png

Uso:
    python generar_matriz_confusion_estructura.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUTA_SALIDA = BASE / "figuras" / "diagramas"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

CLASES = ["sintetica", "real"]

# Intensidades ilustrativas (esquema, no datos reales), pero reflejan el
# patron tipico observado en los experimentos: el modelo distingue "fake"
# muy bien (VP alto) y comete mas errores clasificando "real" (FP > FN).
INTENSIDAD = np.array([
    [0.90, 0.10],
    [0.45, 0.70],
])

ETIQUETAS = np.array([
    ["VP\n(sintetica → sintetica)",  "FN\n(sintetica → real)"],
    ["FP\n(real → sintetica)",  "VN\n(real → real)"],
])

DESCRIPCION = np.array([
    ["Verdadero negativo:\n predijo sintetica y acertó",              "Falso negativo:\nera sintetica y predijo real"],
    ["Falso positivo:\nera real y predijo sintetica", "Verdadero positivo:\n predijo real y acertó"],
])


def main():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    ax.imshow(INTENSIDAD, cmap="Blues", vmin=0, vmax=1)

    n = len(CLASES)
    ax.set_xticks(range(n)); ax.set_xticklabels(CLASES, fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(CLASES, fontsize=11)
    ax.set_xlabel("Predicho", fontsize=12, fontweight="bold")
    ax.set_ylabel("Real", fontsize=12, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for i in range(n):
        for j in range(n):
            color = "white" if INTENSIDAD[i, j] > 0.5 else "black"
            ax.text(j, i - 0.12, ETIQUETAS[i, j], ha="center", va="center",
                     color=color, fontsize=13, fontweight="bold", linespacing=1.4)
            ax.text(j, i + 0.28, DESCRIPCION[i, j], ha="center", va="center",
                     color=color, fontsize=8, linespacing=1.3, style="italic")

    # Cuadricula
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Estructura de la matriz de confusión",
                  fontsize=12, fontweight="bold", pad=45)

    fig.tight_layout()

    salida = RUTA_SALIDA / "matriz_confusion_estructura.png"
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"Guardada: {salida}")


if __name__ == "__main__":
    main()
