# VCT Oddsmaker

**Predicción de probabilidades de victoria en el Valorant Champions Tour (2024–2026)**

Proyecto de Machine Learning sobre el dataset
[Valorant Champion Tour 2021-2026 Data](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data)
(Ryan Luong, licencia MIT, datos scrapeados de vlr.gg).

## Objetivo

Predecir la **probabilidad de victoria de cada equipo en un match / mapa** de VCT,
como lo necesitaría un casino para configurar sus cuotas. Se trabaja únicamente
con las temporadas **2024, 2025 y 2026** (más `all_ids/` para los IDs globales).

## Estructura

```
VCT_Oddsmaker/
├── data/raw/        # dataset descargado (no se sube a git)
│   ├── all_ids/     # IDs globales de jugadores, equipos, torneos y matches
│   ├── vct_2024/
│   ├── vct_2025/
│   └── vct_2026/    # cada año: agents/, ids/, matches/, players_stats/
├── notebooks/       # exploración y experimentos
├── src/             # código fuente
│   └── download_dataset.py
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
python src/download_dataset.py
```

## Notas del dataset

- Usar siempre `Team ID` / `Player ID`, no nombres ni abreviaturas (hay duplicados: dos "Exotic", dos "TP").
- Hay un jugador llamado literalmente `"nan"`: leer CSVs con `keep_default_na=False` si se analiza.
- Nulos en 2k/3k/4k/5k, 1v1..1v5, overtime, duration significan "no ocurrió" → imputar con 0.
- Agentes nuevos aparecen con 0% pick rate antes de su fecha de lanzamiento.
- `Loadout Value` no existe desde Masters Toronto 2025.
- Matches jugados en China tienen muchas columnas de Performance/Economy vacías.
