#!/usr/bin/env python3
"""
mdl_reader.py — Extrae la estructura de un modelo Vensim (.mdl) para leerlo por capas.

Produce, sin interpretar nada:
  - Inventario de variables (tipo, unidades, comentario, ecuación, vista del diagrama).
  - Stocks con sus flujos de entrada y salida.
  - Frontera: variables exógenas (constantes, entradas en función del tiempo) y endógenas.
  - Grafo causal con la polaridad de cada enlace (estimada numéricamente).
  - Bucles de realimentación con su tipo (R = refuerzo, B = equilibrio).
  - Funciones con stocks ocultos (SMOOTH, DELAY...), tablas, datos externos, subíndices.
  - Simulación (si PySD está instalado) para uno o varios escenarios, con resumen por variable.

Uso:
  python mdl_reader.py modelo.mdl --out carpeta_salida
  python mdl_reader.py modelo.mdl --out salida --scenario "Base:" --scenario "Alta:tasa=0.3,meta=50"

Salidas en la carpeta indicada:
  estructura.json   todo lo anterior en formato máquina
  resumen.md        lo mismo en tablas legibles
  sim_<escenario>.csv  trayectorias simuladas (si hubo simulación)

Dependencias: numpy y networkx (obligatorias); pysd (opcional, para simular).
  pip install numpy networkx pysd
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    sys.exit("Falta networkx: pip install networkx")
import numpy as np

CONTROL = {"initial time", "final time", "time step", "saveper"}
DELAY_FUNCS = ["SMOOTH3I", "SMOOTH3", "SMOOTHI", "SMOOTH", "DELAY1I", "DELAY1", "DELAY3I",
               "DELAY3", "DELAY N", "DELAY FIXED", "DELAY", "TREND", "FORECAST", "NPV", "SAMPLE IF TRUE"]
PASS_FUNCS = ["ACTIVE INITIAL", "REINITIAL", "INITIAL"]  # devuelven su primer argumento (salvo en t0)
DATA_FUNCS = ["GET XLS", "GET DIRECT", "GET DATA", "GET 123", "GET VDF", "TIME BASE"]


def norm(s):
    return re.sub(r"[\s_]+", " ", s.strip().strip('"')).lower()


def split_terms(expr):
    """Divide una tasa en términos de primer nivel con su signo: 'a*b - c + d' -> [(+,'a*b'),(-,'c'),(+,'d')]."""
    terms, depth, cur, sign = [], 0, "", 1
    e = expr.strip()
    for i, ch in enumerate(e):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and ch in "+-":
            p = cur.rstrip()
            is_binary = p and p[-1] not in "*/^(,=<>" and not re.search(r"\d[eE]$", p)
            if is_binary or not p:
                if p:
                    terms.append((sign, p.strip()))
                sign = 1 if ch == "+" else -1
                cur = ""
                continue
        cur += ch
    if cur.strip():
        terms.append((sign, cur.strip()))
    return terms


def split_top(s, sep=","):
    """Divide por comas de primer nivel (fuera de paréntesis)."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out]


# ----------------------------------------------------------------- parseo
def parse_mdl(path):
    raw = open(path, encoding="utf-8", errors="replace").read().lstrip("﻿")
    raw = re.sub(r"^\{UTF-8\}\s*", "", raw)
    eq_part, marker, sketch = raw.partition("\\\\\\---///")
    variables, groups, warnings, subs, texts = {}, [], [], [], []
    group = None
    for block in eq_part.split("|"):
        b = block.replace("\\\n", " ").strip()
        if not b:
            continue
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if lines and lines[0].startswith("*"):
            g = [l for l in lines if l.startswith(".")]
            group = g[0][1:].strip() if g else group
            if group:
                groups.append(group)
            continue
        parts = b.split("~")
        eq = parts[0].strip()
        units = parts[1].strip() if len(parts) > 1 else ""
        comment = re.sub(r"\s+", " ", parts[2].strip()) if len(parts) > 2 else ""
        rng = re.search(r"\[([^\]]*)\]\s*$", units)
        units_clean = units[: rng.start()].strip() if rng else units
        kind, lhs, rhs = None, None, None
        if re.search(r":IS:", eq, re.I):
            texts.append(eq.split(":")[0].strip())  # constante de texto (documentación); no es parte de la dinámica
            continue
        if re.match(r"^[^=]*:(?!=)[^=]*$", eq) and "(" not in eq.split(":")[0]:
            dname, _, elems = eq.partition(":")
            elems = elems.split("->")[0]
            el = [e.strip() for e in re.split(r"[,\n]", elems) if e.strip()]
            rng_m = re.match(r"\(\s*([A-Za-z_ ]*?)(\d+)\s*-\s*[A-Za-z_ ]*?(\d+)\s*\)", elems.strip())
            n = (int(rng_m.group(3)) - int(rng_m.group(2)) + 1) if rng_m else len(el)
            subs.append({"nombre": dname.strip(), "elementos": n, "ejemplos": el[:5]})
            continue
        m = re.match(r"^(.*?)(:=|==|=)(.*)$", eq, re.S)
        if m:
            lhs, op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
            if op == ":=":
                kind = "datos"
        else:
            m2 = re.match(r"^([^(]+)\((.*)\)\s*$", eq, re.S)
            if m2:
                lhs, rhs, kind = m2.group(1).strip(), m2.group(2), "tabla"
            elif re.fullmatch(r'[A-Za-z_"][^=()~:]*', eq.strip()):
                lhs, rhs, kind = eq.strip(), "", "datos"
            else:
                warnings.append(f"No se pudo leer la ecuación: {eq[:60]}")
                continue
        dims = []
        if "[" in lhs:
            dims = [d.strip() for d in lhs[lhs.index("[") + 1: lhs.rindex("]")].split(",")]
            lhs = lhs.split("[")[0].strip()
        key = norm(lhs)
        rhs_clean = re.sub(r"\s+", " ", rhs).strip()
        # análisis sobre la ecuación sin subíndices (Births[Towns] -> Births); se conserva la original
        rhs_an = re.sub(r'(?<=[A-Za-z0-9_"\s])\[[^\]()]*\]', "", rhs_clean) if kind != "tabla" else rhs_clean
        valores = None
        if re.fullmatch(r"[-+\d.eE,;\s]+", rhs_an or "x") and re.search(r"[,;]", rhs_an):
            valores = [float(x) for x in re.split(r"[,;\s]+", rhs_an) if x.strip()]
            rhs_an = repr(sum(valores) / len(valores))
        if key in variables and dims:
            variables[key]["definiciones"] = variables[key].get("definiciones", 1) + 1
            continue
        variables[key] = {"nombre": lhs.strip('"'), "ecuacion": rhs_an, "ecuacion_original": rhs_clean,
                          "unidades": units_clean, "comentario": comment, "grupo": group, "tipo": kind,
                          "subindices": dims}
        if valores:
            variables[key]["valores"] = {"n": len(valores), "min": min(valores), "max": max(valores),
                                         "media": sum(valores) / len(valores)}
    views = parse_sketch(sketch) if marker else {}
    m_ds = re.findall(r"^13:(.+)$", sketch, re.M)
    if any(v["tipo"] == "datos" for v in variables.values()):
        names = [v["nombre"] for v in variables.values() if v["tipo"] == "datos"]
        warnings.append("Variables de datos sin ecuación (sus valores vienen de un archivo externo): "
                        + ", ".join(names) + (f". El archivo del modelo apunta al conjunto de datos '{m_ds[0].strip()}'."
                                              if m_ds else "") + " Pasa el archivo con --data para simular.")
    if not marker:
        warnings.append("El archivo no tiene diagrama (sección Sketch): solo hay ecuaciones.")
    if texts:
        warnings.append(f"Se ignoraron {len(texts)} constantes de texto (:IS:), p. ej. '{texts[0]}'; no afectan la dinámica.")
    if subs:
        warnings.append("Subíndices: " + "; ".join(f"{d['nombre']} ({d['elementos']} elementos, p. ej. "
                                                     f"{', '.join(d['ejemplos'][:3])})" for d in subs)
                        + ". La estructura y los bucles se leen una vez por variable (igual para todos los elementos); "
                        "la simulación se resume sumando o promediando los elementos.")
    SUBS_INFO[:] = subs
    return variables, groups, views, warnings


def parse_sketch(sk):
    views, cur = {}, None
    for line in sk.splitlines():
        if line.startswith("*"):
            cur = line[1:].strip()
            views[cur] = {"variables": [], "sombras": []}
        elif cur and line.startswith("10,"):
            f = line.split(",")
            if len(f) > 8:
                name = f[2].strip('"')
                (views[cur]["sombras"] if f[8] == "2" else views[cur]["variables"]).append(name)
        elif line.startswith("///---"):
            break
    return views


# ----------------------------------------------------------- dependencias
def build_name_regex(names):
    pats = []
    for n in sorted(names, key=len, reverse=True):
        words = [re.escape(w) for w in n.split(" ")]
        pats.append(r"[\s_]+".join(words))
    return re.compile(r'(?<![A-Za-z0-9_])"?(' + "|".join(pats) + r')"?(?![A-Za-z0-9_])', re.I)


def classify(variables):
    names = set(variables) | {"time"}
    rx = build_name_regex(names)
    for k, v in variables.items():
        rhs = v["ecuacion"] or ""
        v["deps"] = sorted({norm(m) for m in rx.findall(rhs)} - {k}) if v["tipo"] != "tabla" else []
        v["retardo_oculto"] = [f for f in DELAY_FUNCS if re.search(r"\b" + f.replace(" ", r"\s+") + r"\s*\(", rhs, re.I)]
        if any(re.search(f.replace(" ", r"\s+"), rhs, re.I) for f in DATA_FUNCS):
            v["tipo"] = "datos"
        if v["tipo"]:
            continue
        if re.match(r"^\s*INTEG\s*\(", rhs, re.I):
            inner = rhs[rhs.index("(") + 1: rhs.rindex(")")]
            args = split_top(inner)
            v["tipo"] = "stock"
            v["tasa"] = args[0]
            v["inicial"] = args[1] if len(args) > 1 else ""
            v["deps_tasa"] = sorted({norm(m) for m in rx.findall(args[0])})
            v["deps_inicial"] = sorted({norm(m) for m in rx.findall(v["inicial"])})
            v["terminos"] = []
            for sg, term in split_terms(args[0]):
                t = term.strip()
                while t.startswith("(") and t.endswith(")") and split_top(t[1:-1]) == [t[1:-1].strip()]:
                    t = t[1:-1].strip()
                m = rx.fullmatch(t)
                v["terminos"].append({"signo": sg, "texto": t, "variable": norm(m.group(1)) if m else None})
        elif k in CONTROL:
            v["tipo"] = "control"
        else:
            try:
                float(rhs)
                v["tipo"] = "constante"
            except ValueError:
                v["tipo"] = "auxiliar"
    # flujos = variables que forman por sí solas un término de la tasa de algún stock
    for k, v in variables.items():
        if v["tipo"] == "stock":
            for t in v["terminos"]:
                d = t["variable"]
                if d and d in variables and variables[d]["tipo"] == "auxiliar":
                    variables[d]["tipo"] = "flujo"
    return rx


def build_graph(variables):
    g = nx.DiGraph()
    for k, v in variables.items():
        if v["tipo"] in ("control",):
            continue
        g.add_node(k)
        if v["tipo"] == "stock":
            for d in v["deps_tasa"]:
                g.add_edge(d, k, inicial=False)
            for d in v["deps_inicial"]:
                if d not in v["deps_tasa"]:
                    g.add_edge(d, k, inicial=True)
        else:
            init_only = delay_init_only_deps(v) if v.get("retardo_oculto") else set()
            for d in v["deps"]:
                if d in CONTROL:
                    continue
                g.add_edge(d, k, inicial=d in init_only)
    return g


def delay_init_only_deps(v):
    """Variables que solo aparecen en el argumento de valor inicial de un SMOOTH/DELAY (3.er argumento en adelante):
    fijan el arranque del retardo, no su dinámica."""
    rx = ev_rx[0]
    expr = v["ecuacion"]
    dyn, init = set(), set()
    for f in sorted(DELAY_FUNCS, key=len, reverse=True):
        for m in re.finditer(r"\b" + f.replace(" ", r"\s+") + r"\s*\(", expr, re.I):
            depth, j = 1, m.end()
            while j < len(expr) and depth:
                depth += {"(": 1, ")": -1}.get(expr[j], 0)
                j += 1
            args = split_top(expr[m.end(): j - 1])
            for i, a in enumerate(args):
                names = {norm(x) for x in rx.findall(a)}
                (init if i >= 2 else dyn).update(names)
    outside = {norm(x) for x in rx.findall(re.sub(r"\b(" + "|".join(f.replace(" ", r"\s+") for f in DELAY_FUNCS)
                                                  + r")\s*\([^()]*(\([^()]*\)[^()]*)*\)", "", expr, flags=re.I))}
    return init - dyn - outside


# --------------------------------------------------- evaluación numérica
def to_python(rhs, rx, idx):
    s = rx.sub(lambda m: f"__v{idx[norm(m.group(1))]}__", rhs)
    s = re.sub(r"IF\s+THEN\s+ELSE", "ite", s, flags=re.I)
    s = re.sub(r":AND:", " and ", s, flags=re.I)
    s = re.sub(r":OR:", " or ", s, flags=re.I)
    s = re.sub(r":NOT:", " not ", s, flags=re.I)
    s = s.replace("<>", "!=")
    s = re.sub(r"(?<![<>=!])=(?!=)", "==", s)
    s = s.replace("^", "**")
    for f in sorted(DELAY_FUNCS + PASS_FUNCS, key=len, reverse=True):
        s = re.sub(r"\b" + f.replace(" ", r"\s+") + r"\s*\(", "_passthrough(", s, flags=re.I)
    fmap = {"EXP": "_exp", "LN": "_ln", "LOG": "_log", "SQRT": "_sqrt", "ABS": "abs", "MIN": "min",
            "MAX": "max", "SIN": "_sin", "COS": "_cos", "TAN": "_tan", "INTEGER": "_int",
            "MODULO": "_mod", "STEP": "_step", "PULSE": "_pulse", "RAMP": "_ramp", "XIDZ": "_xidz",
            "ZIDZ": "_zidz"}
    for a, b in fmap.items():
        s = re.sub(r"\b" + a + r"\s*\(", b + "(", s, flags=re.I)
    return re.sub(r"__v(\d+)__", lambda m: f"V[{m.group(1)}]", s)


def make_env(V, time_idx, tables):
    t = lambda: V[time_idx] if time_idx is not None else 0.0
    env = {"ite": lambda c, a, b: a if c else b, "_exp": math.exp, "_ln": math.log,
           "_log": lambda x, b=10: math.log(x, b), "_sqrt": math.sqrt, "_sin": math.sin,
           "_cos": math.cos, "_tan": math.tan, "_int": int, "_mod": lambda a, b: a % b,
           "_step": lambda h, t0: h if t() >= t0 else 0.0,
           "_pulse": lambda s, w: 1.0 if s <= t() < s + max(w, 1e-9) else 0.0,
           "_ramp": lambda sl, a, b=None: sl * (min(t(), b) - a) if t() > a else 0.0,
           "_xidz": lambda a, b, x: a / b if b else x, "_zidz": lambda a, b: a / b if b else 0.0,
           "_passthrough": lambda x, *a: x, "V": V, "abs": abs, "min": min, "max": max}
    return env


def parse_table(body):
    body = re.sub(r"\[[^\]]*\]", "", body)
    pts = [(float(a), float(b)) for a, b in re.findall(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", body)]
    if not pts:
        nums = [float(x) for x in re.findall(r"[-\d.eE+]+", body)]
        h = len(nums) // 2
        pts = list(zip(nums[:h], nums[h:]))
    xs, ys = zip(*sorted(pts)) if pts else ((0,), (0,))
    return lambda x: float(np.interp(x, xs, ys))


def table_points(body):
    body = re.sub(r"\[[^\]]*\]", "", body)
    pts = [(float(a), float(b)) for a, b in re.findall(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", body)]
    return sorted(pts)


def table_shape(pts):
    if len(pts) < 2:
        return "sin puntos suficientes"
    ys = [y for _, y in pts]
    d = np.diff(ys)
    mono = "creciente" if (d >= 0).all() else ("decreciente" if (d <= 0).all() else "sube y baja")
    slopes = [abs((pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0])) for i in range(len(pts) - 1)]
    form = ""
    if mono != "sube y baja" and len(slopes) >= 3:
        if max(slopes) > 1.5 * max(slopes[0], slopes[-1]):
            form = ", forma de S (lenta, rápida, lenta)"
        elif slopes[-1] < slopes[0] * 0.5:
            form = ", se aplana al final (saturación)"
        elif slopes[0] < slopes[-1] * 0.5:
            form = ", empieza plana y se empina (umbral)"
    return f"{mono}{form}; x de {pts[0][0]:g} a {pts[-1][0]:g}, y de {min(ys):g} a {max(ys):g}"


class Evaluator:
    def __init__(self, variables, rx):
        self.keys = sorted(set(variables) | {"time"})
        self.idx = {k: i for i, k in enumerate(self.keys)}
        self.code, self.tables, self.fail = {}, {}, {}
        self.inline_tables, self.wl_fn = {}, {}
        for k, v in variables.items():
            if v["tipo"] == "tabla":
                self.tables[k] = parse_table(v["ecuacion"])
                continue
            expr = v.get("tasa") if v["tipo"] == "stock" else v["ecuacion"]
            wl = re.match(r"^\s*WITH\s+LOOKUP\s*\((.*)\)\s*$", expr or "", re.I | re.S)
            if wl:
                args = split_top(wl.group(1))
                if len(args) == 2:
                    fname = f"_tabla_{len(self.inline_tables) + 1}"
                    fn = parse_table(args[1].strip().strip("()"))
                    self.inline_tables[k] = {"entrada": args[0], "puntos": table_points(args[1])}
                    try:
                        self.code[k] = compile(fname + "(" + to_python(args[0], rx, self.idx) + ")", k, "eval")
                        self.wl_fn[k] = (fname, fn)
                    except Exception as e:  # noqa
                        self.fail[k] = f"no se pudo traducir: {e}"
                    continue
            if not expr or v["tipo"] == "datos" or re.search(r"WITH\s+LOOKUP", expr, re.I):
                self.fail[k] = "ecuación no evaluable automáticamente"
                continue
            try:
                self.code[k] = compile(to_python(expr, rx, self.idx), k, "eval")
            except Exception as e:  # noqa
                self.fail[k] = f"no se pudo traducir: {e}"

    def ev(self, k, V):
        env = make_env(V, self.idx["time"], self.tables)
        # las tablas se llaman como funciones: V[i](x)
        Vc = list(V)
        for tk, fn in self.tables.items():
            Vc[self.idx[tk]] = fn
        for name, fn in self.wl_fn.values():
            env[name] = fn
        env["V"] = Vc
        return float(eval(self.code[k], env))

    def sign(self, src, tgt, V):
        if tgt not in self.code or src not in self.idx:
            return None
        i = self.idx[src]
        if callable(V[i]) or V[i] is None:
            return None
        try:
            base = self.ev(tgt, V)
            h = max(abs(V[i]) * 1e-4, 1e-6)
            Vp = list(V)
            Vp[i] = V[i] + h
            up = self.ev(tgt, Vp)
            Vm = list(V)
            Vm[i] = V[i] - h
            dn = self.ev(tgt, Vm)
        except Exception:  # noqa
            return None
        tol = 1e-10 * max(1.0, abs(base))
        if abs(up - base) > tol and abs(dn - base) > tol and (up - base) * (base - dn) <= 0:
            return "d"  # salto: la variable actúa como selector o umbral (IF, STEP...)
        if abs(up - dn) <= tol and (abs(up - base) > tol or abs(dn - base) > tol):
            return "d"
        d = up - dn
        if abs(d) <= tol:
            return 0
        # un efecto que no crece con el tamaño del empujón es un salto (umbral de un IF), no una pendiente
        try:
            Vp2, Vm2 = list(V), list(V)
            Vp2[i], Vm2[i] = V[i] + 2 * h, V[i] - 2 * h
            d2 = self.ev(tgt, Vp2) - self.ev(tgt, Vm2)
            if abs(d2 - d) <= 1e-6 * abs(d):
                return "d+" if d > 0 else "d-"
        except Exception:  # noqa
            pass
        return 1 if d > 0 else -1


def initial_state(variables, ev, graph, t0):
    """Evalúa todas las variables en el instante inicial, sin simular."""
    V = [None] * len(ev.keys)
    V[ev.idx["time"]] = t0
    for k, v in variables.items():
        if v["tipo"] in ("constante", "control"):
            try:
                V[ev.idx[k]] = float(v["ecuacion"])
            except ValueError:
                pass
    rest = [k for k in variables if V[ev.idx[k]] is None]
    for _ in range(len(rest) + 2):
        progress = False
        for k in rest:
            if V[ev.idx[k]] is not None or k not in variables:
                continue
            v = variables[k]
            try:
                if v["tipo"] == "stock":
                    env = make_env(V, ev.idx["time"], ev.tables)
                    code = to_python(v["inicial"], ev_rx[0], ev.idx)
                    V[ev.idx[k]] = float(eval(code, env))
                elif k in ev.code:
                    V[ev.idx[k]] = ev.ev(k, V)
                else:
                    continue
                progress = True
            except Exception:  # noqa
                continue
        if not progress:
            break
    return V


ev_rx = [None]
SUBS_INFO = []
STUBBED = set()  # variables que no se simularon (función no soportada por PySD)


# ------------------------------------------------------------ simulación
def read_data_file(path):
    """Lee datos externos: formato .DAT de Vensim (nombre y pares tiempo-valor) o CSV/TSV con columna de tiempo."""
    series, cur = {}, None
    if path.lower().endswith((".csv", ".tsv", ".tab", ".txt")):
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python")
        t = df.columns[0]
        for c in df.columns[1:]:
            series[c] = [(float(a), float(b)) for a, b in zip(df[t], df[c]) if str(b) not in ("nan", "")]
        return series
    for line in open(path, encoding="utf-8", errors="replace"):
        nums = re.match(r"^\s*([-+]?\d[\d.eE+-]*)\s+([-+]?\d[\d.eE+-]*)", line)
        if nums:
            if cur is not None:
                series[cur].append((float(nums.group(1)), float(nums.group(2))))
        elif line.strip() and not line.strip().startswith("*"):
            cur = line.strip()
            series[cur] = []
    return {k: v for k, v in series.items() if v}


def _sanitized(raw, stub, data_eq=None):
    eqs, mark, rest = raw.partition("\\\\\\---///")
    out = []
    for b in eqs.split("|"):
        if re.search(r":IS:", b, re.I):
            continue
        head = re.sub(r"^\{UTF-8\}\s*", "", b.split("~")[0].strip())
        if data_eq and norm(head) in data_eq and "=" not in head:
            # variable de datos sin ecuación: se simula como tabla sobre el tiempo con los datos entregados
            b = b.replace(head, f"{head} = {data_eq[norm(head)]}", 1)
        m = re.match(r"(\s*)([^=~]+?)(\s*=)([^~]*)(~.*)", b, re.S)
        if m and norm(m.group(2)) in stub:
            b = m.group(1) + m.group(2) + m.group(3) + " 0\n\t" + m.group(5)
        out.append(b)
    return "|".join(out) + mark + rest


def simulate(path, variables, scenarios, used=frozenset(), data=None):
    try:
        import pysd
    except ImportError:
        return None, "PySD no está instalado: no se simuló (pip install pysd)."
    raw = open(path, encoding="utf-8", errors="replace").read()
    src = os.path.join(os.path.dirname(os.path.abspath(path)), "_pysd_" + os.path.basename(path))
    stub, notes = set(), []
    model = None
    for _ in range(6):
        # copia limpia: sin constantes de texto (:IS:) y con las variables-hoja no simulables en 0
        with open(src, "w", encoding="utf-8") as f:
            f.write(_sanitized(raw, stub, data))
        try:
            model = pysd.read_vensim(src)
            model.run(return_columns=None, final_time=None)
            break
        except Exception as e:  # noqa
            m = re.search(r"[Nn]ot implemented function '([^']+)'", str(e))
            if not m:
                return None, f"PySD no pudo simular el modelo: {str(e)[:200]}"
            fn = m.group(1)
            hits = {k for k, v in variables.items()
                    if re.search(r"\b" + re.escape(fn).replace(r"\ ", r"\s+") + r"\s*\(", v.get("ecuacion") or "", re.I)}
            if not hits or hits & set(used):
                return None, f"PySD no implementa la función {fn.upper()} y la usan variables que afectan al modelo: {sorted(hits)}."
            stub |= hits
            STUBBED.update(hits)
            notes.append(f"PySD no implementa {fn.upper()}: no se simularon {', '.join(variables[h]['nombre'] for h in sorted(hits))} "
                         "(son salidas que no afectan al resto; calcúlalas aparte si las necesitas).")
    if model is None:
        return None, "PySD no pudo simular el modelo."
    out = {}
    cols = [v["nombre"] for k, v in variables.items() if v["tipo"] not in ("tabla",)]
    for i, (name, params) in enumerate(scenarios):
        if i > 0:
            # PySD conserva los parámetros de la corrida anterior: se recarga el modelo en cada escenario
            try:
                model.reload()
            except Exception:  # noqa
                model = pysd.read_vensim(src)
        try:
            res = model.run(params=params, return_columns=cols)
        except Exception:  # noqa
            try:
                res = model.run(params=params)
            except Exception as e:  # noqa
                return out or None, f"Falló la simulación del escenario {name}: {e}"
        out[name] = res
    return out, (" ".join(notes) or None)


def aggregate(res, variables):
    """Columnas 'x[a]' -> una columna 'x' (suma para stocks y flujos, promedio para el resto)."""
    import pandas as pd
    groups = {}
    for c in res.columns:
        if "[" in str(c):
            groups.setdefault(str(c).split("[")[0].strip(), []).append(c)
    if not groups:
        return res, {}
    out, spread = res.copy(), {}
    for base, cols in groups.items():
        v = variables.get(norm(base), {})
        agg = res[cols].sum(axis=1) if v.get("tipo") in ("stock", "flujo") else res[cols].mean(axis=1)
        out[base] = agg
        last = res[cols].iloc[-1]
        spread[base] = {"elementos": len(cols), "operacion": "suma" if v.get("tipo") in ("stock", "flujo") else "promedio",
                        "final_min": [str(last.idxmin()), float(last.min())],
                        "final_max": [str(last.idxmax()), float(last.max())]}
    out = out.drop(columns=[c for cols in groups.values() for c in cols])
    return pd.DataFrame(out), spread


def summarize(res, variables):
    summ = {}
    for k, v in variables.items():
        if v["tipo"] not in ("stock", "flujo", "auxiliar") or v["nombre"] not in res.columns:
            continue
        s = res[v["nombre"]].astype(float)
        if s.max() == s.min() and v["tipo"] != "stock":
            continue
        d = np.diff(s.values)
        mono = "creciente" if (d >= -1e-12).all() else ("decreciente" if (d <= 1e-12).all() else "cambia de dirección")
        summ[v["nombre"]] = {"tipo": v["tipo"], "inicial": round(float(s.iloc[0]), 4),
                             "final": round(float(s.iloc[-1]), 4), "min": round(float(s.min()), 4),
                             "t_min": float(s.idxmin()), "max": round(float(s.max()), 4),
                             "t_max": float(s.idxmax()), "tendencia": mono}
    return summ


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mdl")
    ap.add_argument("--out", default="salida_mdl")
    ap.add_argument("--scenario", action="append", default=[],
                    help='"Nombre:var=valor,var=valor" (repetible). Sin escenarios se simula el caso base.')
    ap.add_argument("--max-loops", type=int, default=300)
    ap.add_argument("--data", action="append", default=[],
                    help="Archivo de datos externos (.DAT de Vensim o CSV con columna de tiempo). Repetible.")
    ap.add_argument("--no-sim", action="store_true", help="No simular (polaridad evaluada solo en el estado inicial)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    variables, groups, views, warnings = parse_mdl(a.mdl)
    rx = classify(variables)
    ev_rx[0] = rx
    graph = build_graph(variables)
    ev = Evaluator(variables, rx)

    # escenarios
    scenarios = []
    for s in a.scenario:
        name, _, rest = s.partition(":")
        params = {}
        for kv in filter(None, rest.split(",")):
            kk, vv = kv.split("=")
            params[kk.strip()] = float(vv)
        scenarios.append((name.strip() or "Base", params))
    if not scenarios:
        scenarios = [("Base", {})]

    used_by_others = {u for u, _ in graph.edges()}
    data_files, data_info = {}, []
    t0c = float(variables.get("initial time", {}).get("ecuacion", 0) or 0)
    t1c = float(variables.get("final time", {}).get("ecuacion", 0) or 0)
    for dpath in a.data:
        try:
            ser = read_data_file(dpath)
        except Exception as e:  # noqa
            warnings.append(f"No se pudo leer el archivo de datos {dpath}: {e}")
            continue
        for name, pts in ser.items():
            key = norm(name)
            if key not in variables:
                warnings.append(f"El archivo {os.path.basename(dpath)} trae la serie '{name}', que no está en el modelo.")
                continue
            ys = [vv for _, vv in pts]
            data_files[key] = ("WITH LOOKUP(Time, ([(" + f"{pts[0][0]:g},{min(ys):g})-({pts[-1][0]:g},{max(ys):g})],"
                               + ",".join(f"({tt:g},{vv:g})" for tt, vv in pts) + "))")
            ts = [tt for tt, _ in pts]
            data_info.append({"variable": variables[key]["nombre"], "archivo": os.path.basename(dpath),
                              "desde": min(ts), "hasta": max(ts), "puntos": len(pts),
                              "primer_valor": pts[0][1], "ultimo_valor": pts[-1][1]})
            if min(ts) > t0c or max(ts) < t1c:
                warnings.append(
                    f"Los datos de '{variables[key]['nombre']}' cubren {min(ts):g}–{max(ts):g}, pero la simulación va de "
                    f"{t0c:g} a {t1c:g}. Fuera de ese rango, la simulación (PySD) mantiene el primer o el último dato "
                    f"({pts[0][1]:g} antes, {pts[-1][1]:g} después). Ese tramo es un supuesto del cálculo, no un dato: "
                    "comprueba cómo lo trata Vensim y dilo en el documento.")
    sims, sim_msg = (None, "Simulación desactivada con --no-sim.") if a.no_sim else \
        simulate(a.mdl, variables, scenarios, used_by_others, data_files)
    if sim_msg:
        warnings.append(sim_msg)
    sims_full, spreads = {}, {}
    if sims:
        for name in list(sims):
            sims_full[name] = sims[name]
            sims[name], spreads[name] = aggregate(sims[name], variables)

    # estados donde evaluar la polaridad
    t0 = float(variables.get("initial time", {}).get("ecuacion", 0) or 0)
    states = []
    if sims:
        for name, res in sims.items():
            params = dict(scenarios)[name] if name in dict(scenarios) else {}
            for t in np.linspace(res.index[0], res.index[-1], 12):
                row = res.loc[res.index[np.abs(res.index - t).argmin()]]
                V = [None] * len(ev.keys)
                V[ev.idx["time"]] = float(row.name)
                for k, v in variables.items():
                    if v["nombre"] in row.index:
                        try:
                            V[ev.idx[k]] = float(row[v["nombre"]])
                        except (TypeError, ValueError):
                            pass
                    elif v["tipo"] == "constante":
                        V[ev.idx[k]] = float(params.get(v["nombre"], v["ecuacion"]))
                states.append(V)
    if not states:
        states = [initial_state(variables, ev, graph, t0)]

    # polaridad de cada enlace
    edges = []
    for u, w, d in graph.edges(data=True):
        if d["inicial"]:
            graph[u][w]["pol"] = "inicial"
            edges.append({"desde": variables.get(u, {}).get("nombre", u), "hacia": variables[w]["nombre"],
                          "polaridad": "inicial", "valor_inicial": True})
            continue
        raw_signs = [s for s in (ev.sign(u, w, V) for V in states) if s is not None]
        jumps = {s for s in raw_signs if s in ("d+", "d-")}
        signs = [("d" if s in ("d+", "d-") else s) for s in raw_signs]
        nz = {s for s in signs if s not in (0, "d")}
        if variables.get(u, {}).get("tipo") == "tabla":
            pol = "tabla"  # función tabulada: el signo depende de la forma de la tabla
        elif not signs:
            pol = "?"
        elif not nz and len(jumps) == 1:
            pol = "+ (salto)" if jumps == {"d+"} else "- (salto)"  # umbral con sentido definido
        elif not nz and "d" in signs:
            pol = "salto"  # selector o umbral: no tiene un signo continuo
        elif not nz and variables[w]["retardo_oculto"]:
            pol = "tiempo de retardo"  # parámetro de un SMOOTH/DELAY: cambia la velocidad, no el sentido
        elif not nz:
            pol = "0"   # sin efecto en los estados evaluados (p. ej. selector discreto)
        elif len(nz) == 1:
            pol = "+" if nz.pop() > 0 else "-"
        else:
            pol = "±"
        graph[u][w]["pol"] = pol
        edges.append({"desde": variables.get(u, {}).get("nombre", u), "hacia": variables[w]["nombre"],
                      "polaridad": pol, "valor_inicial": d["inicial"]})

    # variables que dependen de stocks pero no cambian en ninguna simulación, y stocks con trayectorias idénticas
    inert, dup = set(), []
    g_dyn = nx.DiGraph([(u, w) for u, w, d in graph.edges(data=True) if not d["inicial"]])
    from_stocks = set()
    for k, v in variables.items():
        if v["tipo"] == "stock" and k in g_dyn:
            from_stocks |= nx.descendants(g_dyn, k)
    if sims:
        for k, v in variables.items():
            if v["tipo"] in ("auxiliar", "flujo") and k in from_stocks and k not in STUBBED:
                ok = True
                for res in sims.values():
                    if v["nombre"] not in res.columns:
                        ok = False
                        break
                    sr = res[v["nombre"]].astype(float)
                    if (sr.max() - sr.min()) > 1e-9 * max(1.0, abs(sr.mean())):
                        ok = False
                        break
                if ok:
                    inert.add(k)
        stock_names = [v["nombre"] for v in variables.values() if v["tipo"] == "stock"]
        for i, a1 in enumerate(stock_names):
            for a2 in stock_names[i + 1:]:
                same = True
                for res in sims.values():
                    if a1 not in res.columns or a2 not in res.columns:
                        same = False
                        break
                    x, y = res[a1].astype(float).values, res[a2].astype(float).values
                    if np.max(np.abs(x - y)) > 1e-6 * max(1.0, np.max(np.abs(x))):
                        same = False
                        break
                if same:
                    dup.append((a1, a2))
    for k in sorted(inert):
        warnings.append(f"'{variables[k]['nombre']}' depende de stocks pero no cambia en ninguna simulación "
                        "(p. ej. una suma que se conserva): los bucles que pasan por ella no actúan.")
    for a1, a2 in dup:
        warnings.append(f"Los stocks '{a1}' y '{a2}' tienen trayectorias idénticas: posible duplicado del mismo concepto.")

    # bucles
    dyn = nx.DiGraph([(u, w, d) for u, w, d in graph.edges(data=True) if not d["inicial"]])
    loops = []
    try:
        cyc = nx.simple_cycles(dyn, length_bound=40)
    except TypeError:
        cyc = nx.simple_cycles(dyn)
    for c in cyc:
        if len(loops) >= a.max_loops:
            warnings.append(f"Se alcanzó el límite de {a.max_loops} bucles; puede haber más.")
            break
        pols = [dyn[c[i]][c[(i + 1) % len(c)]].get("pol", "?") for i in range(len(c))]
        base = [p.split(" ")[0] for p in pols]
        if "?" in base or "0" in base or "salto" in base:
            tipo = "?"
        elif "±" in base:
            tipo = "depende del estado"
        else:
            tipo = "R" if base.count("-") % 2 == 0 else "B"
        start = min(range(len(c)), key=lambda i: (variables[c[i]]["tipo"] != "stock", c[i]))
        c = c[start:] + c[:start]
        pols = pols[start:] + pols[:start]
        loops.append({"recorrido": [variables[n]["nombre"] for n in c], "polaridades": pols,
                      "negativos": sum(p.startswith("-") for p in pols), "tipo": tipo, "longitud": len(c),
                      "umbral": any("salto" in p for p in pols),
                      "stocks": [variables[n]["nombre"] for n in c if variables[n]["tipo"] == "stock"],
                      "retardos_ocultos": [variables[n]["nombre"] for n in c if variables[n]["retardo_oculto"]],
                      "inactivo": any(n in inert for n in c)})
    loops.sort(key=lambda l: (l["longitud"], l["recorrido"]))
    # conjunto independiente de bucles más cortos (heurística tipo SILS, Oliva 2004):
    # se recorren los bucles de menor a mayor y se conserva uno si aporta al menos un enlace nuevo
    seen_edges, sils = set(), []
    for l in loops:
        names = [norm(n) for n in l["recorrido"]]
        es = {(names[i], names[(i + 1) % len(names)]) for i in range(len(names))}
        if l["inactivo"]:
            continue
        if es - seen_edges:
            sils.append(l)
            seen_edges |= es
            l["esencial"] = True
    # familias: bucles con el mismo tipo y los mismos stocks son variantes de un mismo mecanismo
    fams = {}
    for l in loops:
        fams.setdefault((l["tipo"], tuple(sorted(l["stocks"])), l["inactivo"]), []).append(l)
    families = sorted(fams.values(), key=lambda ls: (len(ls[0]["stocks"]), ls[0]["longitud"]))
    cnt = {"R": 0, "B": 0}
    fam_out = []
    for ls in families:
        t = ls[0]["tipo"]
        if t in cnt:
            cnt[t] += 1
            code = f"{t}{cnt[t]}"
        else:
            code = "?" + str(len(fam_out) + 1)
        for j, l in enumerate(ls, 1):
            l["codigo"] = code if len(ls) == 1 else f"{code}.{j}"
            l["familia"] = code
        fam_out.append({"codigo": code, "tipo": t, "stocks": ls[0]["stocks"], "variantes": len(ls),
                        "inactiva": ls[0]["inactivo"],
                        "representante": ls[0]["recorrido"], "polaridades": ls[0]["polaridades"],
                        "retardos_ocultos": sorted({r for l in ls for r in l["retardos_ocultos"]})})

    # frontera: endógeno = alcanzable desde algún stock
    stocks = [k for k, v in variables.items() if v["tipo"] == "stock"]
    endo = set()
    for s in stocks:
        endo |= nx.descendants(dyn, s) if s in dyn else set()
        endo.add(s)
    view_of = defaultdict(list)
    for vname, vv in views.items():
        for n in vv["variables"]:
            view_of[norm(n)].append(vname)
    used = {u for u, _ in graph.edges()}
    inventory = []
    for k, v in sorted(variables.items(), key=lambda kv: (kv[1]["tipo"] or "", kv[0])):
        if v["tipo"] == "control":
            continue
        item = {"nombre": v["nombre"] + (f"[{','.join(v['subindices'])}]" if v.get("subindices") else ""),
                "tipo": v["tipo"], "unidades": v["unidades"],
                "comentario": v["comentario"], "grupo": v["grupo"],
                "ecuacion": (f"{v['valores']['n']} valores (mín {v['valores']['min']:g}, máx {v['valores']['max']:g}, "
                             f"media {v['valores']['media']:g})" if v.get("valores")
                             else v.get("ecuacion_original", v["ecuacion"])),
                "endogena": k in endo, "vistas": view_of.get(k, []),
                "usada_por_otras": k in used or v["tipo"] == "stock"}
        if v["retardo_oculto"]:
            item["retardo_oculto"] = v["retardo_oculto"]
        if v["tipo"] == "stock":
            item["entradas"], item["salidas"] = [], []
            for t in v["terminos"]:
                label = variables[t["variable"]]["nombre"] if t["variable"] in variables else f"[{t['texto']}]"
                (item["entradas"] if t["signo"] > 0 else item["salidas"]).append(label)
            if any(not t["variable"] for t in v["terminos"]):
                item["tasa_compuesta"] = v["tasa"]
            item["valor_inicial"] = v["inicial"]
        if k in ev.fail:
            item["aviso"] = ev.fail[k]
        inventory.append(item)
        if views and v["tipo"] not in ("tabla",) and not view_of.get(k) and v["tipo"] != "control":
            warnings.append(f"'{v['nombre']}' tiene ecuación pero no aparece en ninguna vista del diagrama.")
        if not item["usada_por_otras"] and v["tipo"] in ("constante", "auxiliar"):
            warnings.append(f"'{v['nombre']}' no la usa ninguna otra variable (solo salida o sobrante).")

    tablas = []
    for k, v in variables.items():
        pts, entrada = None, None
        if v["tipo"] == "tabla":
            pts = table_points(v["ecuacion"])
        elif k in ev.inline_tables:
            pts, entrada = ev.inline_tables[k]["puntos"], ev.inline_tables[k]["entrada"]
        if not pts:
            continue
        tablas.append({"variable": v["nombre"], "entrada": entrada, "puntos": pts, "forma": table_shape(pts)})
        if entrada and sims:
            lo, hi = pts[0][0], pts[-1][0]
            code = compile(to_python(entrada, rx, ev.idx), "entrada", "eval")
            for name, res in sims.items():
                xs = []
                for tt, row in res.iterrows():
                    Vr = [None] * len(ev.keys)
                    Vr[ev.idx["time"]] = float(tt)
                    for kk, vv in variables.items():
                        if vv["nombre"] in row.index:
                            try:
                                Vr[ev.idx[kk]] = float(row[vv["nombre"]])
                            except (TypeError, ValueError):
                                pass
                        elif vv["tipo"] == "constante":
                            Vr[ev.idx[kk]] = float(dict(scenarios).get(name, {}).get(vv["nombre"], vv["ecuacion"]))
                    try:
                        xs.append((float(tt), float(eval(code, make_env(Vr, ev.idx["time"], ev.tables)))))
                    except Exception:  # noqa
                        pass
                out = [(tt, x) for tt, x in xs if x < lo - 1e-9 or x > hi + 1e-9]
                if out:
                    warnings.append(f"En el escenario '{name}', la entrada de la tabla '{v['nombre']}' sale de su rango "
                                    f"[{lo:g}, {hi:g}] desde t={out[0][0]:g} (llega a {max(x for _, x in xs):g}, "
                                    f"mínimo {min(x for _, x in xs):g}). Vensim mantiene el valor del extremo: desde ahí "
                                    "el efecto queda fijo por cómo se dibujó la tabla, no por el fenómeno.")

    control = {k: variables[k]["ecuacion"] for k in CONTROL if k in variables}
    result = {"archivo": os.path.basename(a.mdl), "control_tiempo": control,
              "unidad_tiempo": variables.get("final time", {}).get("unidades", ""),
              "grupos": groups, "vistas": views, "variables": inventory, "enlaces": edges,
              "bucles_esenciales": [l["codigo"] for l in sils], "subindices": SUBS_INFO, "familias_de_bucles": fam_out, "tablas": tablas, "datos_externos": data_info, "bucles": loops, "avisos": warnings,
              "conteo": {t: sum(1 for i in inventory if i["tipo"] == t)
                         for t in ("stock", "flujo", "auxiliar", "constante", "tabla", "datos")},
              "polaridad_metodo": "Derivada numérica de cada ecuación respecto a cada entrada, evaluada en "
                                  + (f"{len(states)} estados de la simulación" if sims else "el estado inicial")
                                  + ". '±' = el signo cambia según el estado; 'salto' = actúa como selector o umbral "
                                  "(IF THEN ELSE, STEP); '+ (salto)' o '- (salto)' = umbral con sentido definido: el efecto "
                                  "ocurre de golpe al cruzar un valor; '0' = sin efecto en esos estados; 'inicial' = solo fija el "
                                  "valor de partida de un stock; 'tabla' = función tabulada (ver su forma); 'tiempo de retardo' = "
                                  "parámetro de un SMOOTH/DELAY; '?' = no evaluable (revisar a mano)."}
    if sims:
        result["simulacion"] = {}
        for name, res in sims.items():
            res.to_csv(os.path.join(a.out, f"sim_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.csv"))
            if spreads.get(name):
                sims_full[name].to_csv(os.path.join(a.out, f"sim_{re.sub(r'[^A-Za-z0-9]+', '_', name)}_por_elemento.csv"))
            result["simulacion"][name] = {"parametros": dict(scenarios).get(name, {}),
                                          "resumen": summarize(res, variables),
                                          "por_elemento": spreads.get(name, {})}
    with open(os.path.join(a.out, "estructura.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    write_md(result, os.path.join(a.out, "resumen.md"))
    print(f"Listo: {a.out}/estructura.json y {a.out}/resumen.md "
          f"({result['conteo']['stock']} stocks, {len(loops)} bucles, {len(warnings)} avisos)")


def write_md(r, path):
    L = [f"# Estructura de {r['archivo']}", ""]
    c = r["control_tiempo"]
    L.append(f"Horizonte: {c.get('initial time','?')} a {c.get('final time','?')} {r['unidad_tiempo']} · "
             f"paso {c.get('time step','?')} · conteo: {r['conteo']}")
    if r["vistas"]:
        L.append("Vistas del diagrama: " + ", ".join(r["vistas"]))
    L += ["", "## Stocks", "", "| Stock | Unidades | Entradas | Salidas | Valor inicial | Comentario |", "|---|---|---|---|---|---|"]
    for v in r["variables"]:
        if v["tipo"] == "stock":
            L.append(f"| {v['nombre']} | {v['unidades']} | {', '.join(v['entradas'])} | {', '.join(v['salidas'])} | "
                     f"{v['valor_inicial']} | {v['comentario']} |")
    L += ["", "## Variables", "", "| Variable | Tipo | Endógena | Unidades | Ecuación | Comentario |", "|---|---|---|---|---|---|"]
    for v in r["variables"]:
        if v["tipo"] != "stock":
            eq = v["ecuacion"].replace("|", "/")
            L.append(f"| {v['nombre']} | {v['tipo']} | {'sí' if v['endogena'] else 'no'} | {v['unidades']} | `{eq}` | {v['comentario']} |")
    L += ["", "## Enlaces y polaridad", "", r["polaridad_metodo"], "", "| Desde | Hacia | Polaridad | Solo valor inicial |", "|---|---|---|---|"]
    for e in r["enlaces"]:
        L.append(f"| {e['desde']} | {e['hacia']} | {e['polaridad']} | {'sí' if e['valor_inicial'] else ''} |")
    fams = r["familias_de_bucles"]
    L += ["", f"## Familias de bucles ({len(fams)} familias, {len(r['bucles'])} bucles)", "",
          "Una familia agrupa los bucles del mismo tipo que pasan por los mismos stocks: son variantes de un mismo "
          "mecanismo (por ejemplo, el mismo efecto por dos caminos paralelos). Explica familias, no bucles sueltos.", "",
          "| Familia | Tipo | Stocks | Variantes | Activa en la simulación | Recorrido más corto |", "|---|---|---|---|---|---|"]
    for f in fams:
        rec = " → ".join(f"{n} ({p})" for n, p in zip(f["representante"], f["polaridades"])) + f" → {f['representante'][0]}"
        L.append(f"| {f['codigo']} | {f['tipo']} | {', '.join(f['stocks'])} | {f['variantes']} | {'no' if f['inactiva'] else 'sí'} | {rec} |")
    ess = [l for l in r["bucles"] if l.get("esencial")]
    L += ["", f"## Bucles esenciales ({len(ess)} de {len(r['bucles'])})", "",
          "Conjunto independiente de bucles más cortos: cada uno aporta al menos un enlace que ningún bucle más corto "
          "contiene. Juntos cubren la realimentación activa del modelo (se excluyen los bucles que pasan por variables "
          "que nunca cambian). Empieza la lectura por aquí.", "",
          "| Código | Tipo | Stocks | Grupos | Recorrido |", "|---|---|---|---|---|"]
    grp = {v["nombre"]: (v.get("grupo") or "sin grupo") for v in r["variables"]}
    for l in ess:
        rec = " → ".join(f"{n} ({p})" for n, p in zip(l["recorrido"], l["polaridades"])) + f" → {l['recorrido'][0]}"
        gs = sorted({grp.get(n, "sin grupo") for n in l["recorrido"]})
        L.append(f"| {l['codigo']} | {l['tipo']} | {', '.join(l['stocks'])} | {', '.join(gs)} | {rec} |")
    show = r["bucles"] if len(r["bucles"]) <= 40 else []
    L += ["", f"## Todos los bucles ({len(r['bucles'])}{'; la lista completa está en estructura.json' if not show else ''})", "",
          "| Código | Recorrido | Negativos | Tipo | Stocks | Retardos ocultos |", "|---|---|---|---|---|---|"]
    for l in show:
        rec = " → ".join(f"{n} ({p})" for n, p in zip(l["recorrido"], l["polaridades"])) + f" → {l['recorrido'][0]}"
        L.append(f"| {l['codigo']} | {rec} | {l['negativos']} | {l['tipo']} | {', '.join(l['stocks'])} | "
                 f"{', '.join(l['retardos_ocultos'])} |")
    if r.get("datos_externos"):
        L += ["", "## Datos externos", "", "| Variable | Archivo | Desde | Hasta | Puntos | Primer valor | Último valor |",
              "|---|---|---|---|---|---|---|"]
        for d in r["datos_externos"]:
            L.append(f"| {d['variable']} | {d['archivo']} | {d['desde']:g} | {d['hasta']:g} | {d['puntos']} | "
                     f"{d['primer_valor']:g} | {d['ultimo_valor']:g} |")
    if r.get("tablas"):
        L += ["", "## Tablas (relaciones definidas por puntos)", "",
              "| Variable | Entrada | Forma | Puntos |", "|---|---|---|---|"]
        for t in r["tablas"]:
            pts = ", ".join(f"({x:g}, {y:g})" for x, y in t["puntos"])
            L.append(f"| {t['variable']} | `{t['entrada'] or 'se usa como función: nombre(x)'}` | {t['forma']} | {pts} |")
    hid = [v for v in r["variables"] if v.get("retardo_oculto")]
    if hid:
        L += ["", "## Retardos ocultos (cada uno contiene uno o más stocks internos)", ""]
        L += [f"- {v['nombre']}: {', '.join(v['retardo_oculto'])}" for v in hid]
    if "simulacion" in r:
        for name, s in r["simulacion"].items():
            L += ["", f"## Simulación: {name} {s['parametros'] or ''}", "",
                  "| Variable | Tipo | Inicial | Final | Mín (t) | Máx (t) | Tendencia |", "|---|---|---|---|---|---|---|"]
            if s.get("por_elemento"):
                L.insert(len(L) - 2, "Variables con subíndices: " + "; ".join(
                    f"{b} = {d['operacion']} de {d['elementos']} elementos (al final: mín {d['final_min'][0]} = "
                    f"{d['final_min'][1]:.4g}, máx {d['final_max'][0]} = {d['final_max'][1]:.4g})"
                    for b, d in s["por_elemento"].items()) + "\n")
            for vn, x in s["resumen"].items():
                L.append(f"| {vn} | {x['tipo']} | {x['inicial']} | {x['final']} | {x['min']} ({x['t_min']:g}) | "
                         f"{x['max']} ({x['t_max']:g}) | {x['tendencia']} |")
    L += ["", "## Avisos", ""] + [f"- {w}" for w in r["avisos"]] if r["avisos"] else []
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
