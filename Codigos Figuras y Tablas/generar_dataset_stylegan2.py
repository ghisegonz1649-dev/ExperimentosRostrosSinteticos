"""
generar_stylegan2.py
Genera imágenes de rostros con StyleGAN2 usando los pesos OFICIALES de NVIDIA
(modelo preentrenado en FFHQ, resolución 1024x1024).

Objetivo: producir 7,000 imágenes de StyleGAN2 auténtico, con procedencia clara
y reproducible, para reemplazar el dataset ruidoso de Kaggle. Las imágenes se
generan a 1024x1024; después tu crear_particiones.py las reduce a 256x256 con
Lanczos (nunca se amplían píxeles).

Reproducibilidad: cada imagen se genera a partir de una semilla fija (0, 1, 2...),
así que este script produce EXACTAMENTE las mismas 7,000 caras cada vez que se
ejecuta. El truncamiento psi=0.8 coincide con el usado en tu dataset de StyleGAN3
(troykueh), para mantener condiciones comparables entre ambos generadores GAN.

--------------------------------------------------------------------------------
REQUISITOS PREVIOS (una sola vez):

1) Instala PyTorch con CUDA (ver requirements_stylegan2.txt) y luego:
       pip install -r requirements_stylegan2.txt

2) Clona el repositorio oficial de NVIDIA en la misma carpeta que este script:
       git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

   Debe quedar así:
       tu_carpeta/
           generar_stylegan2.py        <- este script
           stylegan2-ada-pytorch/      <- el repo clonado
               dnnlib/
               torch_utils/
               legacy.py
               ...

3) Ejecuta:
       python generar_stylegan2.py

--------------------------------------------------------------------------------
"""
import os
import sys

import numpy as np
import PIL.Image
import torch

# --- Hacer visible el repositorio oficial de NVIDIA ---
RUTA_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stylegan2-ada-pytorch")
if not os.path.isdir(RUTA_REPO):
    sys.exit("ERROR: no encuentro la carpeta 'stylegan2-ada-pytorch'. "
             "Clónala con:\n  git clone "
             "https://github.com/NVlabs/stylegan2-ada-pytorch.git")
sys.path.insert(0, RUTA_REPO)

import dnnlib   # noqa: E402  (del repo de NVIDIA)
import legacy   # noqa: E402  (del repo de NVIDIA)

# ------------------------------- Configuración -------------------------------

# Pesos oficiales de NVIDIA: StyleGAN2-ADA 
URL_MODELO = ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/"
              "pretrained/ffhq.pkl")

N_IMAGENES = 7000        
TRUNCATION_PSI = 0.8      
SEMILLA_INICIAL = 0       
CARPETA_SALIDA = "StyleGAN2"   

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------- Main -----------------------------------
def main() -> None:
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    print(f"Dispositivo: {DISPOSITIVO}")
    print(f"Descargando/cargando pesos oficiales de NVIDIA (FFHQ)...")

    with dnnlib.util.open_url(URL_MODELO) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(DISPOSITIVO)
    G.eval()

    etiqueta = torch.zeros([1, G.c_dim], device=DISPOSITIVO)

    print(f"Generando {N_IMAGENES} imágenes (psi={TRUNCATION_PSI})...\n")
    generadas = 0
    for i in range(N_IMAGENES):
        semilla = SEMILLA_INICIAL + i
        ruta = os.path.join(CARPETA_SALIDA, f"stylegan2_{semilla:05d}.png")
        if os.path.exists(ruta):
            continue
        z = torch.from_numpy(
            np.random.RandomState(semilla).randn(1, G.z_dim)
        ).to(DISPOSITIVO)

        with torch.no_grad():
            img = G(z, etiqueta,
                    truncation_psi=TRUNCATION_PSI,
                    noise_mode="const")  
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128)
        img = img.clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(ruta)

        generadas += 1
        if generadas % 100 == 0:
            print(f"  {i + 1}/{N_IMAGENES} generadas...")

    print(f"\nListo. Imágenes en la carpeta: {CARPETA_SALIDA}")
    print("Siguiente paso: mueve esta carpeta a "
          "Datasets/StyleGAN2/ (reemplazando la anterior) y corre "
          "crear_particiones.py")

if __name__ == "__main__":
    main()
