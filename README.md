# VCT Oddsmaker

**Predicción de probabilidades de victoria en el Valorant Champions Tour (2024–2026)**

Proyecto de Machine Learning sobre el dataset
[Valorant Champion Tour 2021-2026 Data](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)
(Ryan Luong, licencia MIT, datos scrapeados de vlr.gg).

## Objetivo

Predecir la **probabilidad de victoria de cada equipo en un match / mapa** de VCT,
como lo necesitaría un casino para configurar sus cuotas. Se trabaja únicamente
con las temporadas **2024, 2025 y 2026** (más `all_ids/` para los IDs globales).

## Primera entrega: EDA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANDRIX089/VCT_Oddsmaker/blob/main/notebooks/01_EDA.ipynb)

Notebook: [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb). Funciona directamente en Google Colab:
descarga `data/processed/maps_dataset.csv` desde este repositorio (no necesita Kaggle ni credenciales).

## Estructura

```
VCT_Oddsmaker/
├── data/
│   ├── raw/                     # dataset de Kaggle (no se sube a git, ~137 MB)
│   │   ├── all_ids/             # IDs globales de jugadores, equipos, torneos y matches
│   │   ├── vct_2024/ vct_2025/ vct_2026/   # cada año: agents/, ids/, matches/, players_stats/
│   └── processed/
│       └── maps_dataset.csv     # 1 fila por mapa (3,267 x 80), listo para el EDA / modelado
├── notebooks/
│   └── 01_EDA.ipynb             # análisis exploratorio (primera entrega)
├── src/
│   ├── download_dataset.py      # descarga el dataset de Kaggle (2024-2026)
│   └── build_dataset.py         # une las tablas y genera data/processed/maps_dataset.csv
├── requirements.txt
└── README.md
```

## Reproducir en local

```bash
pip install -r requirements.txt
python src/download_dataset.py     # descarga data/raw/
python src/build_dataset.py        # genera data/processed/maps_dataset.csv
jupyter notebook notebooks/01_EDA.ipynb
```

## Variable objetivo

`team_a_wins` en `maps_dataset.csv`: 1 si el Team A gana el mapa, 0 si gana el Team B.
Clases balanceadas (50.3% / 49.7%). Ver el diccionario de columnas en el notebook.

## Notas del dataset

- Usar siempre `Team ID` / `Player ID`, no nombres ni abreviaturas (hay duplicados: dos "Exotic", dos "TP").
- Hay un jugador llamado literalmente `"nan"`: leer CSVs con `keep_default_na=False` si se analiza.
- Nulos en 2k/3k/4k/5k, 1v1..1v5, overtime, duration significan "no ocurrió" → imputar con 0.
- Agentes nuevos aparecen con 0% pick rate antes de su fecha de lanzamiento.
- `Loadout Value` no existe desde Masters Toronto 2025.
- Matches jugados en China tienen muchas columnas de Performance/Economy vacías.
