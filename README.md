# VCT Oddsmaker

**Predicción de probabilidades de victoria por mapa en el Valorant Champions Tour (2024–2026)**

*Aprendizaje de Máquina · Grupo C1 · Periodo 2026-2 · Profesora Alejandra Moreno*

---

## Presentación del proyecto

VCT Oddsmaker estima, **antes de que empiece un mapa**, la probabilidad de que cada equipo lo gane en el circuito profesional de VALORANT (Valorant Champions Tour). Nos ponemos en el lugar del equipo de *data science* de una casa de apuestas: a partir de esa probabilidad se fija la cuota que se le ofrece al público. Por ejemplo, si el modelo dice que el equipo A gana con probabilidad 0.65, la cuota justa es 1 / 0.65 ≈ 1.54.

El proyecto se desarrolló en tres entregas:

1. **Análisis exploratorio (EDA):** construcción del dataset de un mapa por fila, calidad de datos y primeros patrones.
2. **Clasificación binaria:** ¿quién gana el mapa? Comparación de cuatro clasificadores con variables *pre-match*.
3. **Regresión y multiclase:** ¿por cuántas rondas gana? y ¿quién gana y por cuánto?

## Motivación

- **Un problema real con dinero en juego.** Las casas de apuestas publican cuotas para cada mapa de VALORANT profesional. Un error sistemático en esas probabilidades se traduce directamente en pérdidas, así que no basta con adivinar el ganador: las probabilidades tienen que estar bien calibradas.
- **Por qué Machine Learning.** Fijar cuotas a ojo es subjetivo y no escala a cientos de mapas por temporada. Una regla simple ("gana el de mejor ranking") ignora el mapa, la forma reciente y quién eligió el mapa. Un modelo combina muchas variables, aprende interacciones, se puede reentrenar con cada resultado nuevo y su error se puede medir (log-loss, Brier, calibración).
- **Un reto de datos interesante.** El dataset original es relacional (131 archivos CSV), no tiene columna de fecha y gran parte de las estadísticas se registran *durante* el mapa, lo que obliga a cuidar la **fuga de información** (*data leakage*): solo se puede predecir con lo que se sabía antes de empezar.

## Integrantes

| Integrante | Código |
|---|---|
| Ivan Daniel Villamizar Torres | 2252001 |
| Kevin Adrian Romero Lozada | 2251502 |
| Santiago Cortez Mojica | 2251513 |

---

## Dataset

### Fuente y alcance

- **Fuente:** [Valorant Champion Tour 2021-2026 Data](https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data) (Kaggle, autor Ryan Luong, licencia MIT). Los datos fueron *scrapeados* de [vlr.gg](https://www.vlr.gg), la referencia estadística de VALORANT competitivo.
- **Alcance elegido:** temporadas **2024, 2025 y 2026**, el formato actual de ligas regionales más torneos internacionales (más `all_ids/` para los identificadores globales).
- **Dataset original:** 131 archivos CSV, 1,314 columnas y 1.36 GB, organizados por año en `agents/`, `ids/`, `matches/` y `players_stats/`.
- **Unidad de análisis:** **un mapa jugado**. Un match Bo3/Bo5 se compone de varios mapas, así que trabajar por mapa da más datos (3,267 mapas frente a ~1,280 matches), conserva información propia de cada mapa (el mapa en sí, quién lo eligió) y la probabilidad de la serie se puede derivar de la de cada mapa.

### Construcción de la tabla

`src/build_dataset.py` une las tablas de resultados, estadísticas de jugadores, economía, fase de picks/bans y métodos de victoria de ronda, y genera **`data/processed/maps_dataset.csv`**: **3,267 mapas × 80 columnas**, un mapa por fila.

| Dimensión | Valores |
|---|---|
| Mapas por temporada | 2024: 1,104 · 2025: 1,277 · 2026: 886 |
| Regiones | China 703 · Pacific 694 · EMEA 687 · Americas 674 · International 509 |
| Mapas jugados | Lotus 538 · Haven 368 · Split 362 · Bind 342 · Ascent 290 · Icebox 285 · Sunset 267 · Pearl 209 · Breeze 199 · Fracture 145 · Abyss 144 · Corrode 118 |
| Quién eligió el mapa | Team A 41.4 % · Team B 41.0 % · *decider* 17.7 % |
| Fase | Liga/Grupos 1,800 · Playoffs/Bracket 1,433 · Otro 34 |
| Volumen | 68 equipos · 40 torneos · 1,279 matches |

### Diccionario de columnas principales

| Columna | Descripción |
|---|---|
| `year`, `region`, `tournament`, `stage`, `stage_group`, `match_type` | Contexto del mapa (región: Americas / EMEA / Pacific / China / International) |
| `match_id`, `game_id` | IDs de vlr.gg, crecientes en el tiempo: se usan para ordenar cronológicamente porque no hay columna de fecha |
| `map` | Mapa jugado (Ascent, Bind, Haven, …) |
| `map_picked_by` | Quién eligió el mapa en el draft: `A`, `B` o `decider` (mapa restante) |
| `team_a`, `team_b`, `team_a_id`, `team_b_id` | Equipos (Team A = primer equipo listado en vlr.gg, no el favorito) |
| `score_a`, `score_b`, `round_diff`, `total_rounds`, `overtime`, `duration_sec` | Resultado del mapa |
| **`team_a_wins`** | **Variable objetivo de clasificación:** 1 si Team A ganó el mapa, 0 si ganó Team B |
| `atk_score_*`, `def_score_*` | Rondas ganadas atacando / defendiendo |
| `rating_*`, `acs_*`, `kills_*`, `deaths_*`, `assists_*`, `kast_*`, `adr_*`, `hs_pct_*`, `first_kills_*`, `first_deaths_*` | Promedio de los 5 jugadores del equipo en ese mapa (sufijo `_a` / `_b`) |
| `pistol_won_*`, `eco_won_*`, `semi_eco_won_*`, `semi_buy_won_*`, `full_buy_won_*` | Rondas ganadas por tipo de economía |
| `win_elim_*`, `win_detonated_*`, `win_defused_*`, `win_time_*` | Rondas ganadas por método (eliminación, spike, defusa, tiempo) |
| `*_diff` | Diferencia Team A − Team B de cada estadística |

### Variables objetivo

- **`team_a_wins`** (clasificación binaria): clases balanceadas, **50.3 % / 49.7 %**, por lo que no se necesita re-muestreo y una moneda al aire da 50 %.
- **`round_diff = score_a − score_b`** (regresión): media 0.19, desviación estándar 5.93 rondas. Es bimodal: no existen valores entre −1 y +1 porque siempre hay ganador. El 34.6 % de los mapas se decide por 3 rondas o menos.
- **4 clases por margen** (multiclase): B gana cómodo (≤ −5), B gana ajustado (−4 a −2), A gana ajustado (+2 a +4) y A gana cómodo (≥ +5). El corte de 5 rondas es la mediana de |`round_diff`| y coincide con el hándicap −4.5 de las casas de apuestas.

### Calidad de datos

| Grupo de columnas | % nulos | Causa | Tratamiento |
|---|---|---|---|
| `ot_score_a/b` | ~90 % | El mapa no llegó a overtime | Ya está en `overtime` (0/1) |
| Economía (`pistol_won_*`, `eco_won_*`, …) | ~24 % | vlr.gg no publica economía de los matches de China | Imputación o indicador de faltante |
| `rating_*`, `kast_*` | ~12 % | Post-match incompleto (mayoría China) | `acs` y `kills` sí están y sirven de *proxy* |
| `adr_*`, `hs_pct_*`, `first_kills_*` | ~8 % | Igual | Imputación |
| `duration_sec` | ~12 % | No disponible en vlr.gg | No se usa como variable (solo se conoce al terminar) |

Los valores atípicos son partidos reales (palizas 13-2, overtimes largos) y **se conservan**. Ningún nulo afecta a la variable objetivo ni al contexto (`map`, `region`, `map_picked_by`).

### Hallazgos del EDA que guiaron el modelado

- Las estadísticas del propio mapa (rating, ACS, KAST, ADR) explican casi por completo quién gana, pero son *post-match*: usarlas sería **fuga de información**. Por eso los modelos solo usan **versiones históricas** de ellas.
- Hay señal *pre-match* legítima: el equipo que elige el mapa lo gana el **53.2 %** de las veces, y la regla "apostar al de mejor historial" acierta el **55.6 %**.
- Las rondas pistola y los *first kills* pesan mucho en el resultado.
- Hay alta multicolinealidad entre las estadísticas de rendimiento (> 0.85), lo que obliga a seleccionar o regularizar.

---

## Notebooks del proyecto

| Notebook | Contenido | Abrir |
|---|---|---|
| [`01_EDA.ipynb`](notebooks/01_EDA.ipynb) | Análisis exploratorio de datos (primera entrega) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANDRIX089/VCT_Oddsmaker/blob/main/notebooks/01_EDA.ipynb) |
| [`02_Modelos_Clasificacion.ipynb`](notebooks/02_Modelos_Clasificacion.ipynb) | Modelos de clasificación: ¿quién gana el mapa? (segunda entrega) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANDRIX089/VCT_Oddsmaker/blob/main/notebooks/02_Modelos_Clasificacion.ipynb) |
| [`03_Regresion_Multiclase.ipynb`](notebooks/03_Regresion_Multiclase.ipynb) | Modelos de regresión del margen de rondas y clasificación multiclase (entrega final) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANDRIX089/VCT_Oddsmaker/blob/main/notebooks/03_Regresion_Multiclase.ipynb) |

Los tres notebooks funcionan directamente en Google Colab: descargan `data/processed/maps_dataset.csv` desde este repositorio, sin Kaggle ni credenciales.

---

## Variables pre-match y protocolo de evaluación

Todos los modelos usan **solo información disponible antes del mapa**, calculada con `shift(1)` sobre el orden cronológico. Todas las variables son diferencias Team A − Team B:

- **`elo_diff`:** rating Elo secuencial, leído antes de procesar el resultado del mapa.
- **Win rate:** histórico (`wr_all_diff`), de los últimos 10 mapas (`wr_10_diff`) y en ese mapa (`map_wr_diff`), suavizados hacia 0.5.
- **Rendimiento reciente** (últimos 10 mapas): `rd_10_diff`, `acs_10_diff`, `rating_10_diff`, `pistol_10_diff` y `fk_10_diff`.
- **Contexto:** `picker` (quién eligió el mapa), `hist_min`, playoffs, región y mapa.

**Partición temporal:** train = primer 75 % de 2024–2025, validación = último 25 % de 2024–2025, test = toda la temporada 2026. El test se usa **una sola vez**, al final. Los hiperparámetros se eligen con validación más validación cruzada temporal (`TimeSeriesSplit`). Cada prueba queda registrada con qué se movió, por qué y qué pasó.

## Modelos utilizados

### Clasificación (¿quién gana el mapa?) — `02_Modelos_Clasificacion.ipynb`

| Modelo | Descripción breve |
|---|---|
| **Gaussian Naive Bayes** | Aplica el teorema de Bayes suponiendo que cada variable sigue una distribución normal dentro de cada clase y que las variables son independientes entre sí. Es muy rápido y da probabilidades directamente; su hiperparámetro principal es `var_smoothing`. |
| **Árbol de decisión** | Divide los datos con preguntas del tipo "¿`elo_diff` > 4?" hasta formar hojas, y predice la clase más frecuente de cada hoja. Es interpretable, pero sobreajusta si no se limita su profundidad (`max_depth`) o el mínimo de mapas por hoja (`min_samples_leaf`). |
| **Random Forest** | Promedia muchos árboles entrenados con muestras distintas de los datos y subconjuntos de variables, para reducir la varianza de un árbol solo. Se ajustan el número de árboles, la profundidad, el tamaño de las hojas y `max_features`. |
| **SVM** (*Support Vector Machine*) | Busca la frontera que separa las clases con el mayor margen posible. Con kernels (RBF, polinómico) puede trazar fronteras no lineales. `C` controla cuánto se castigan los errores y `gamma` el alcance de cada punto. |

**Resultado:** los cuatro quedan entre 57 % y 59 % de accuracy en 2026. El mejor es Gaussian NB (59.1 %), al nivel de la regla "gana el de mayor Elo" (59.4 %). Los **dos mejores**, elegidos con el protocolo, son **Gaussian NB y Random Forest**.

### Regresión (¿por cuántas rondas gana?) — `03_Regresion_Multiclase.ipynb`

La variable objetivo pasa de `team_a_wins` (0/1) a **`round_diff`**, un resultado real y continuo cuyo signo dice quién ganó y cuyo tamaño dice por cuánto. Las variables de entrada son las mismas.

| Modelo | Descripción breve |
|---|---|
| **Árbol de decisión regresor** | Igual que el clasificador, pero cada hoja predice el **promedio** del margen de sus mapas, así que la predicción es una "escalera". Se ajustan `max_depth`, `min_samples_leaf`, `criterion` y la poda `ccp_alpha`. |
| **Random Forest regresor** | Promedia muchos árboles regresores. Hojas grandes (muchos mapas por promedio) reducen el ruido y el sobreajuste. |
| **SVR** (*Support Vector Regression*) | Busca una función lo más suave posible que deje a la mayoría de los mapas dentro de un "tubo" de ancho ε rondas; solo se penalizan los que quedan fuera. Se ajustan `C`, `gamma`, `epsilon` y el kernel. |

**Resultado:** RMSE ≈ 5.87 rondas y R² ≈ 0.025 en 2026. Los **dos mejores** son **Random Forest y SVR**. Además, el margen predicho se convierte en probabilidad de victoria con P(gana A) = Φ(ŷ/σ), lo que permite cotizar también el hándicap.

### Clasificación multiclase (¿quién gana y por cuánto?)

Se usan los mismos cuatro clasificadores sobre las 4 clases por margen, eligiendo con F1-macro. Los **dos mejores** son **Random Forest y Gaussian NB**, con ≈ 31 % de accuracy frente a 25 % del azar. No superan a una regla simple basada en el Elo, y las victorias "ajustadas" resultan casi impredecibles.

### Conclusión general

Las tres formas de plantear el problema llegan al mismo techo, el que marca el Elo. Con estadísticas del propio mapa se superaría el 95 %, pero eso es fuga de información: **el límite no está en los modelos, sino en la poca información que existe antes de que empiece un mapa.**

---

## Estructura del repositorio

```
VCT_Oddsmaker/
├── data/
│   ├── raw/                     # dataset de Kaggle (no se sube a git, ~137 MB)
│   │   ├── all_ids/             # IDs globales de jugadores, equipos, torneos y matches
│   │   ├── vct_2024/ vct_2025/ vct_2026/   # cada año: agents/, ids/, matches/, players_stats/
│   └── processed/
│       └── maps_dataset.csv     # 1 fila por mapa (3,267 x 80), listo para el EDA / modelado
├── notebooks/
│   ├── 01_EDA.ipynb                    # análisis exploratorio (primera entrega)
│   ├── 02_Modelos_Clasificacion.ipynb  # GNB, árbol, Random Forest y SVM (segunda entrega)
│   └── 03_Regresion_Multiclase.ipynb   # árbol, RF y SVR regresores + multiclase (entrega final)
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
jupyter notebook notebooks/        # abrir 01_EDA, 02_Modelos_Clasificacion o 03_Regresion_Multiclase
```

## Notas del dataset

- Usar siempre `Team ID` / `Player ID`, no nombres ni abreviaturas (hay duplicados: dos "Exotic", dos "TP").
- Hay un jugador llamado literalmente `"nan"`: leer CSVs con `keep_default_na=False` si se analiza.
- Nulos en 2k/3k/4k/5k, 1v1..1v5, overtime, duration significan "no ocurrió" → imputar con 0.
- Agentes nuevos aparecen con 0% pick rate antes de su fecha de lanzamiento.
- `Loadout Value` no existe desde Masters Toronto 2025.
- Matches jugados en China tienen muchas columnas de Performance/Economy vacías.
