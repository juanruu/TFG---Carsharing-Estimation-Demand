"""
apply_cleaning_patches.py
=========================

One-shot script that patches both predictive notebooks
(`xgboost_solution.ipynb` and `lstm_dataset_creation.ipynb`) so that
they share a single, consistent cleaning pipeline implemented in
``trip_cleaning.py``.

Run once from the repository root:

    python apply_cleaning_patches.py

The script is idempotent: re-running it leaves already-patched
notebooks unchanged. A backup is created next to each notebook with
the suffix ``.bak`` before the first modification.
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path

import nbformat
from nbformat import NotebookNode

REPO_ROOT  = Path(__file__).resolve().parent
NB_LSTM    = REPO_ROOT / "lstm_dataset_creation.ipynb"
NB_XGBOOST = REPO_ROOT / "xgboost_solution.ipynb"


# ---------------------------------------------------------------------------
# New cell sources
# ---------------------------------------------------------------------------

LSTM_NEW_AUX_SOURCE = '''\
# Cleaning compartido entre los frameworks XGBoost y LSTM
# (definicion completa en trip_cleaning.py).
from trip_cleaning import clean_trips, parse_date_utc

CLEAN_STATS = []   # acumulador de auditoria de cleaning por ciudad


def coord_to_h3(lat: float, lon: float, resolution: int = H3_RESOLUTION) -> str:
    """Convierte coordenada a celda H3."""
    return h3.latlng_to_cell(lat, lon, resolution)


def load_city(city: str) -> pd.DataFrame:
    """
    Carga, LIMPIA y procesa los registros de una ciudad.

    El cleaning compartido se ejecuta antes de cualquier asignacion espacial
    para que ambos pipelines (XGBoost y LSTM) operen sobre datos depurados
    bajo identicos criterios. Devuelve un DataFrame con
    (city, h3_cell, datetime_utc, date, hour, day_of_week, month) listo
    para la agregacion de demanda.
    """
    path = DATA_DIR / f"{city}_trips.txt"
    df = pd.read_csv(path, sep=" ", quotechar='"')

    # 1. Cleaning compartido: filtra outliers de distancia, duracion y geo
    df, stats = clean_trips(df, city)
    CLEAN_STATS.append(stats)

    # 2. Parseo de fecha a UTC
    print(f"  Parseando fechas ({city})...")
    df["datetime_utc"] = df["s_date"].apply(parse_date_utc)
    df = df.dropna(subset=["datetime_utc"])

    # 3. Asignacion H3 (lat/lon ya provienen del cleaning)
    print(f"  Asignando celdas H3 ({city})...")
    df["h3_cell"] = df.apply(
        lambda r: coord_to_h3(r["lat"], r["lon"]), axis=1
    )

    # 4. Componentes temporales
    df["date"]        = df["datetime_utc"].dt.date
    df["hour"]        = df["datetime_utc"].dt.hour
    df["day_of_week"] = df["datetime_utc"].dt.dayofweek   # 0 = lunes
    df["month"]       = df["datetime_utc"].dt.month
    df["city"]        = city

    return df[["city", "h3_cell", "datetime_utc", "date",
               "hour", "day_of_week", "month"]]


print("Funciones definidas.")
'''


LSTM_AUDIT_SOURCE = '''\
# Auditoria del cleaning multi-ciudad
# (resume cuantos viajes se descartan por filtro y ciudad)
import pandas as pd
df_audit = pd.DataFrame(CLEAN_STATS).set_index("city")
print("\\n" + "=" * 70)
print("AUDITORIA DEL CLEANING")
print("=" * 70)
print(df_audit.to_string())
'''


LSTM_AUDIT_HEADER = '''\
## 3.1 Auditoria del cleaning compartido

Resumen del numero de viajes descartados por cada filtro de
``trip_cleaning.py`` (distancia <100 m, duracion fuera del rango 1-480 min,
outliers geograficos y duplicados exactos). Estas estadisticas se utilizan
en la seccion 3.1.2 del TFG para justificar los umbrales adoptados.
'''


XGBOOST_NEW_CLEAN_SOURCE = '''\
# Limpieza compartida entre frameworks (XGBoost + LSTM)
# El detalle de los criterios y umbrales esta en trip_cleaning.py
import sys
sys.path.insert(0, ".")
from trip_cleaning import clean_trips, parse_date_local

# 1. Cleaning unificado (filtra outliers de distancia, duracion y geografia)
milano_trips, stats = clean_trips(milano_trips, "milano")
print(f"\\nResumen del cleaning Milan: {stats}")

# 2. Parseo de fechas en hora local (XGBoost trabaja en hora local)
milano_trips["s_date"] = milano_trips["s_date"].apply(parse_date_local)
milano_trips["e_date"] = milano_trips["e_date"].apply(parse_date_local)
milano_trips = milano_trips.dropna(subset=["s_date"])

# 3. Seleccion de columnas (incluye las derivadas que produce clean_trips)
columns_to_keep = [
    "vin", "traveltime", "google_time", "google_distance",
    "s_date", "s_coord", "s_fuel", "e_date", "e_coord", "e_fuel",
    "dist_km", "dur_min", "lat", "lon",
]
milano_trips = milano_trips[columns_to_keep].copy()

# 4. Feature engineering temporal
milano_trips["hour"]        = milano_trips["s_date"].dt.hour
milano_trips["day_name"]    = milano_trips["s_date"].dt.day_name()
milano_trips["day_of_week"] = milano_trips["s_date"].dt.dayofweek
milano_trips["month"]       = milano_trips["s_date"].dt.month

# 5. Geometria a partir de lat/lon ya parseadas (no hace falta re-split)
from shapely.geometry import Point
milano_trips["geometry"] = milano_trips.apply(
    lambda r: Point(r["lon"], r["lat"]), axis=1
)
gdf_trips = gpd.GeoDataFrame(milano_trips, geometry="geometry", crs="EPSG:4326")
'''


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def backup(path: Path) -> None:
    """Create a *.bak copy of *path* if it does not yet exist."""
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup -> {bak.name}")


def find_cell_index(nb: NotebookNode, predicate) -> int:
    """Return the index of the first cell that satisfies *predicate*, or -1."""
    for i, c in enumerate(nb.cells):
        if predicate(c):
            return i
    return -1


def make_code_cell(source: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(source=source)


def make_markdown_cell(source: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source)


# ---------------------------------------------------------------------------
# Notebook patchers
# ---------------------------------------------------------------------------

def patch_lstm_notebook(path: Path) -> None:
    print(f"[{path.name}]")
    nb = nbformat.read(path, as_version=4)

    # 1. Replace the auxiliary-functions cell (the one containing parse_date / load_city)
    idx_aux = find_cell_index(
        nb,
        lambda c: c.cell_type == "code" and "def load_city" in c.source,
    )
    if idx_aux < 0:
        print("  WARNING: load_city cell not found, skipping aux replacement.")
    else:
        if "from trip_cleaning import" in nb.cells[idx_aux].source:
            print("  load_city cell already patched, skipping.")
        else:
            nb.cells[idx_aux] = make_code_cell(LSTM_NEW_AUX_SOURCE)
            print("  load_city cell replaced.")

    # 2. Insert the cleaning-audit cells right after the city-loading cell
    idx_load = find_cell_index(
        nb,
        lambda c: c.cell_type == "code" and "for city in CITIES" in c.source
                  and "load_city" in c.source,
    )
    if idx_load < 0:
        print("  WARNING: city-loading cell not found, skipping audit insertion.")
    else:
        already_audited = any(
            "AUDITORIA DEL CLEANING" in (c.source or "") for c in nb.cells
        )
        if already_audited:
            print("  audit cells already present, skipping.")
        else:
            nb.cells.insert(idx_load + 1, make_markdown_cell(LSTM_AUDIT_HEADER))
            nb.cells.insert(idx_load + 2, make_code_cell(LSTM_AUDIT_SOURCE))
            print("  audit cells inserted.")

    nbformat.write(nb, path)
    print(f"  saved -> {path.name}")


def patch_xgboost_notebook(path: Path) -> None:
    print(f"[{path.name}]")
    nb = nbformat.read(path, as_version=4)

    # The XGBoost notebook has the cleaning split across two cells:
    #   (a) date parsing  (contains 'Normalize and parse date columns')
    #   (b) column selection + clean_distance/clean_time + dist_km filter
    # We collapse both into a single cell that uses trip_cleaning.

    idx_date = find_cell_index(
        nb,
        lambda c: c.cell_type == "code"
                  and "Normalize and parse date columns" in c.source,
    )
    idx_clean = find_cell_index(
        nb,
        lambda c: c.cell_type == "code"
                  and "def clean_distance" in c.source,
    )

    if idx_date < 0 or idx_clean < 0:
        print("  WARNING: original cleaning cells not found; nothing to patch.")
        return

    if "from trip_cleaning import" in nb.cells[idx_date].source:
        print("  XGBoost notebook already patched, skipping.")
        return

    # Replace the first (date) cell with the consolidated cleaning cell,
    # and clear the second (clean_distance) cell -- removing it would shift
    # all downstream indices and is unnecessary.
    nb.cells[idx_date] = make_code_cell(XGBOOST_NEW_CLEAN_SOURCE)
    nb.cells[idx_clean] = make_code_cell(
        "# Celda fusionada con la anterior (limpieza centralizada en "
        "trip_cleaning.clean_trips). Mantenida vacia para no desplazar "
        "los indices del notebook."
    )
    print("  cleaning cells consolidated into trip_cleaning.clean_trips.")

    nbformat.write(nb, path)
    print(f"  saved -> {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not NB_LSTM.exists():
        raise FileNotFoundError(NB_LSTM)
    if not NB_XGBOOST.exists():
        raise FileNotFoundError(NB_XGBOOST)

    print(f"Repository root: {REPO_ROOT}\n")

    backup(NB_LSTM)
    patch_lstm_notebook(NB_LSTM)
    print()

    backup(NB_XGBOOST)
    patch_xgboost_notebook(NB_XGBOOST)


if __name__ == "__main__":
    main()
