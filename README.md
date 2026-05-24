# Reproducción Experimental y Evaluación de Modelos Avanzados para la Segmentación de Meningiomas

Este repositorio contiene el código fuente y el marco experimental desarrollado como Trabajo de Fin de Grado (TFG). Su objetivo es proporcionar un entorno estructurado, modular y reproducible para la comparación de arquitecturas en la tarea de segmentación 3D de tumores cerebrales (Meningiomas), utilizando los datos oficiales del reto BraTS-MEN 2023.

* **Autor:** Diego Jesús Torrejón Cabrera.
* **Tutor:** Javier Sánchez Pérez.
* **Institución:** Escuela de Ingeniería Informática, Universidad de Las Palmas de Gran Canaria (ULPGC).
* **Titulación:** Grado en Ciencia e Ingeniería de Datos.
* **Curso:** 2025/2026.

## Arquitecturas Evaluadas
* **U-Net 3D:** Arquitectura base.
* **SegResNet:** Reducción de dimensionalidad con bloques residuales y diseño asimétrico.
* **Swin UNETR:** Arquitectura basada en Transformers.
* **SegMamba:** Modelado de espacio de estados.
* **SegMamba-V2:** Mejora reciente del modelo **SegMamba**.

## Características del Marco Experimental
* **Análisis Exploratorio de Datos (EDA):** Scripts para visualizar distribuciones demográficas, estadísticas de resolución geométrica y análisis topológico de las subregiones tumorales.
* **Procesamiento de Imágenes Médicas:** Utilidades basadas en `MONAI` para carga segura, transformaciones sobre los datos y almacenamiento en caché.
* **Entrenamiento:** Bucle de entrenamiento en PyTorch con Precisión Mixta Automática, optimizador Adam y función de pérdida DiceLoss.
* **Evaluación:** Inferencia por ventana deslizante calculando métricas estandarizadas (Dice, IoU, HD95) para clases aisladas (NETC, SNFH, ET) y regiones clínicas (WT, TC, ET).
* **Visualización Científica:** Generación de gráficos, incluyendo curvas de aprendizaje, resultados de evaluación y análisis de eficiencia de VRAM frente a tiempo y precisión.

## Estructura del Repositorio

    ├── configs/
    │   └── config.py              
    ├── data/                      # Volúmenes NIfTI y metadatos clínicos
    │   ├── brats-train-val-2023/  
    │   └── splits.json            
    ├── data_analysis/             # Salida: Archivos resumen y exportaciones de EDA
    ├── graphs/                    # Salida: Gráficos generados
    ├── logs/                      # Salida: Historiales de entrenamiento y métricas
    ├── src/
    │   ├── data/                  # Lógica de carga y manipulación de datos
    │   │   ├── dataset.py         
    │   │   ├── make_splits.py     
    │   │   └── transforms.py      
    │   ├── data_analysis/         # Scripts de exploración de datos, demografía y dimensiones
    │   │   ├── analyze_subregions.py
    │   │   ├── check_dimensions.py
    │   │   ├── generates_3d_planes.py
    │   │   ├── labels.py
    │   │   ├── modalities_visualization.py
    │   │   ├── plot_clinical_data.py
    │   │   └── resolution_summary.py
    │   ├── models/                # Código fuente de las arquitecturas neuronales
    │   │   ├── segmamba.py
    │   │   ├── segmambav2.py
    │   │   ├── segresnet.py
    │   │   ├── swinunetr.py
    │   │   └── unet.py
    │   ├── utils/                 # Herramientas transversales y librerías de soporte
    │   │   ├── clinical_io.py     
    │   │   ├── image_processing.py
    │   │   ├── nifti_io.py        
    │   │   └── plot_utils.py      
    │   └── visualization/         # Scripts para exportar resultados visuales
    │       ├── export_inference.py
    │       ├── test_graphics.py
    │       └── train_graphics.py
    ├── evaluate.py                # Ejecución de evaluación multirregional en test
    ├── train.py                   # Bucle principal de optimización y entrenamiento
    ├── requirements.txt           
    ├── LICENSE                    
    └── README.md                  

## Requisitos
* Python 3.10
* GPU compatible con CUDA, los experimentos se ejecutaron con una GPU NVIDIA RTX 3080 (10 GB VRAM) con CUDA 12.1
* Windows Subsystem for Linux (WSL) sobre Windows

## Instalación

### 1. **Clonar repositorio**

Primero se debe clonar este repositorio y luego acceder a la carpeta del proyecto. Se recomienda crear y activar un entorno virtual.

```bash
git clone https://github.com/Tiopipi/TFG-meningioma-segmentaion
cd TFG-meningioma-segmentaion
```

### 2. **Conjunto de datos**

Este repositorio no incluye el conjunto de datos por restricciones de peso. Para replicar el entorno experimental, se debe realizar los siguientes pasos:
1. **Descarga:** Solicitar acceso y descargar el conjunto de datos oficial del reto **BraTS-MEN 2023** (tarea de Meningiomas Intracraneales).
2. **Estructura:** Crear una carpeta llamada `data/` en la raíz de este repositorio y organizar los datos siguiendo la siguiente estructura, necesaria para que `configs/config.py` localice los archivos correctamente:

```text
    ├── data/
    │   ├── brats-train-val-2023/
    │   │   ├── BraTS-MEN-Train/                  # Casos de entrenamiento con máscaras
    │   │   ├── BraTS-MEN-Validation/             # Casos de validación del reto
    │   │   └── Meningioma supplementary... .xlsx # Archivo clínico complementario
    │   └── splits.json                           # Generado mediante 'src/data/make_splits.py'
```

### 3. **Instalar PyTorch**

Una vez descargados los datos, descargar la versión de PyTorch con soporte para la tarjeta gráfica que se vaya a utilizar, preferiblemente para CUDA 12.1.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
```

### 4. **Instalar requirements.txt**

Instalar el resto de dependencias.

```bash
pip install -r requirements.txt
```

## Configuración

Antes de ejecutar los experimentos, se recomienda revisar la configuración establecida en `configs/config.py` por si se necesita cambiar algún parámetro o ajustar el nombre de las carpetas.

## Uso

### Ejecución de scripts para el análisis de datos

Para ejecutar los scripts relacionados con el análisis exploratorio de los datos:

```bash
python src/data_analysis/check_dimensions.py
python src/data_analysis/plot_clinical_data.py
python src/data_analysis/modalities_visualization.py
...
```

### Ejecución del experimento

1. **Generar partición de los datos**

Crear el archivo splits.json que define los conjuntos de entrenamiento, validación y prueba de forma reproducible

```bash
python src/data/make_splits.py
```

2. **Ejecutar archivo de entrenamiento**

Antes de lanzar el entrenamiento del modelo, se debe modificar el parámetro `model_name` en el bloque `if __name__ == "__main__":` de `train.py` para seleccionar la arquitectura deseada.

```bash
python train.py
```

3. **Ejecutar código de evaluación**

Tras finalizar el entrenamiento, para calcular las métricas de rendimiento (Dice, IoU, HD95) sobre el conjunto de prueba y generar los reportes CSV:

```bash
python evaluate.py
```

### Visualización de los resultados

Para obtener los resultados visuales y exportar las predicciones de los modelos entrenados:

```bash
# Generar gráficos comparativos y de rendimiento
python src/visualization/train_graphics.py
python src/visualization/test_graphics.py

# Exportar inferencias NIfTI de pacientes de prueba
python src/visualization/export_inference.py
```