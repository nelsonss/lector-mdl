"""Pruebas mínimas: el analizador lee los modelos de ejemplo y su simulación coincide con PySD."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pysd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
LECTOR = RAIZ / "mdl_reader.py"
EJEMPLOS = RAIZ / "ejemplos"


def correr_lector(mdl: Path, salida: Path, *escenarios: str) -> dict:
    args = [sys.executable, str(LECTOR), str(mdl), "--out", str(salida)]
    for e in escenarios:
        args += ["--scenario", e]
    subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads((salida / "estructura.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("modelo,stocks", [("Convergencia_a_valor.mdl", 1), ("Deuda_Tecnica_v2_1.mdl", 7)])
def test_encuentra_los_stocks(tmp_path, modelo, stocks):
    estructura = correr_lector(EJEMPLOS / modelo, tmp_path, "Base:")
    assert estructura["conteo"]["stock"] == stocks


def test_bucles_esenciales_y_tipos_coherentes(tmp_path):
    estructura = correr_lector(EJEMPLOS / "Deuda_Tecnica_v2_1.mdl", tmp_path, "Base:")
    assert estructura["bucles_esenciales"], "no hay bucles esenciales"
    for bucle in estructura["bucles"]:
        if bucle["tipo"] in ("R", "B"):
            assert (bucle["negativos"] % 2 == 0) == (bucle["tipo"] == "R"), bucle["codigo"]


def test_simulacion_coincide_con_pysd(tmp_path):
    correr_lector(EJEMPLOS / "Deuda_Tecnica_v2_1.mdl", tmp_path, "Base:", "Fijo:POLITICA DE REFACTOR=1")
    modelo = pysd.read_vensim(str(EJEMPLOS / "Deuda_Tecnica_v2_1.mdl"))
    for nombre, params in [("Base", {}), ("Fijo", {"POLITICA DE REFACTOR": 1})]:
        lector = pd.read_csv(tmp_path / f"sim_{nombre}.csv")["Funcionalidades Pendientes"].iloc[-1]
        directo = modelo.run(params=params, return_columns=["Funcionalidades Pendientes"], reload=True).iloc[-1, 0]
        assert lector == pytest.approx(directo, rel=1e-6), nombre
