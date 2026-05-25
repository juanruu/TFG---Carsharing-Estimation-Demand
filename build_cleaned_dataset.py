"""
build_cleaned_dataset.py
========================

Genera el dataset limpio compartido por ambas alternativas de modelado
(XGBoost y LSTM). Lee los ficheros crudos de las 10 ciudades, aplica la
pipeline de limpieza centralizada (trip_cleaning.clean_trips) y guarda
el resultado en ``datasets/trips_cleaned.parquet``.

Ejecución::

    python build_cleaned_dataset.py
"""

from pathlib import Path

import pandas as pd

from trip_cleaning import clean_trips

DATA_DIR = Path("cs_datasets")
OUTPUT_DIR = Path("datasets")
OUTPUT_PATH = OUTPUT_DIR / "trips_cleaned.parquet"

CITIES = [
    "amsterdam", "berlin", "firenze", "kobenhavn",
    "milano", "muenchen", "roma", "stockholm", "torino", "wien",
]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    frames = []
    audit_rows = []

    for city in CITIES:
        path = DATA_DIR / f"{city}_trips.txt"
        print(f"\n[{city}] Cargando {path} ...")
        df = pd.read_csv(path, sep=" ", quotechar='"')

        df, stats = clean_trips(df, city)
        audit_rows.append(stats)

        df["city"] = city
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    df_all.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nDataset limpio guardado en {OUTPUT_PATH}")
    print(f"  Filas totales: {len(df_all):,}")
    print(f"  Ciudades:      {sorted(df_all['city'].unique())}")
    print(f"  Tamaño:        {OUTPUT_PATH.stat().st_size / 1024**2:.1f} MB")

    df_audit = pd.DataFrame(audit_rows).set_index("city")
    print("\n" + "=" * 70)
    print("AUDITORÍA DEL CLEANING")
    print("=" * 70)
    print(df_audit.to_string())


if __name__ == "__main__":
    main()
