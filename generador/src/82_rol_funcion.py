"""Hay estructura rol -> tipo de funcion en los datos, o es una intuicion?

La idea a comprobar: en una app de venta de libros el VENDEDOR pide funciones de
administracion y el COMPRADOR funciones de compra. Si eso es real, deberia verse en
los verbos: cada familia de rol deberia concentrarse en unos verbos distintos.

Se mide sobre los dos corpus que tenemos con rol explicito:
  - Dalpiaz: 1.680 historias escritas por personas en 22 proyectos reales
  - el profesor v3: backlogs sinteticos con el sector y el plantel anotados

La prueba es de independencia: si el verbo no dependiera del rol, la distribucion de
verbos seria la misma para todas las familias. Se reporta V de Cramer sobre la tabla
rol x verbo y los verbos mas caracteristicos de cada familia por razon de frecuencia.

  python src/82_rol_funcion.py

out: results/82_rol_funcion.csv
"""
import collections
import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_HIST = re.compile(
    r"^\s*as\s+(?:an?|the)?\s*(?P<rol>[^,]{2,40}?)\s*,\s*i(?:'m| am)?\s*"
    r"(?:want|need|wish|would like|am able|can|'d like)\s*(?:to\s+|a\s+|able\s+to\s+)?"
    r"(?P<medio>.{3,})$", re.I)
# familias de rol, por nucleo de la palabra: es la hipotesis a falsear, no el resultado
FAMILIAS = {
    "administra": ["admin", "administrator", "manager", "owner", "moderator", "officer",
                   "supervisor", "principal", "coordinator", "operator"],
    "consume": ["customer", "shopper", "buyer", "client", "user", "visitor", "guest",
                "reader", "viewer", "member", "patient", "student", "attendee", "rider",
                "traveller", "traveler", "subscriber"],
    "opera": ["staff", "employee", "agent", "clerk", "picker", "courier", "driver",
              "cleaner", "technician", "teacher", "trainer", "cook", "waiter", "nurse",
              "doctor", "pharmacist", "mechanic", "stylist"],
    "publica": ["seller", "vendor", "publisher", "author", "artist", "supplier",
                "provider", "host", "creator", "narrator"],
}
NUCLEO = {n: f for f, ns in FAMILIAS.items() for n in ns}
PARADA = {"the", "a", "an", "my", "their", "all", "some", "this", "that", "to", "be",
          "have", "get", "make", "do", "able"}


def familia(rol):
    """La familia del rol. NO basta con la ultima palabra: "Staff member" tiene por
    nucleo "member" (consumidor) cuando es claramente personal de la organizacion, y
    con esa regla el 3,8 % de las historias humanas parecia mal clasificado cuando el
    error era mio. Si alguna palabra del rol es de una familia de TRABAJO, esa manda
    sobre la de consumo."""
    p = [w for w in re.findall(r"[a-z]+", (rol or "").lower())]
    hallados = [NUCLEO[w] for w in p if w in NUCLEO]
    if not hallados:
        return None
    trabajo = [f for f in hallados if f != "consume"]
    return trabajo[0] if trabajo else hallados[-1]


def verbo(medio):
    """primer verbo del medio, saltando auxiliares y determinantes"""
    p = [w for w in re.findall(r"[a-z]+", (medio or "").lower())]
    for w in p:
        if w not in PARADA:
            return w
    return None


def pares_dalpiaz():
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_dalpiaz.parquet")
    out = []
    for lst in d.tareas:
        for h in list(lst):
            m = RE_HIST.match(re.sub(r"\s+", " ", str(h)))
            if m:
                out.append((m.group("rol"), m.group("medio")))
    return out


def pares_profesor():
    p = f"{BASE}/data/synth/backlogs_v3_qwen2.5_7b.jsonl"
    out = []
    for l in open(p, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        for h in r.get("historias", []) or []:
            # el profesor v3 guarda cada historia como dict {historia, tareas}
            txt = h.get("historia", "") if isinstance(h, dict) else str(h)
            m = RE_HIST.match(re.sub(r"\s+", " ", txt))
            if m:
                out.append((m.group("rol"), m.group("medio")))
    return out


def cramer(tab):
    T = tab.values.astype(float)
    n = T.sum()
    if n == 0:
        return 0.0
    fila = T.sum(1, keepdims=True)
    col = T.sum(0, keepdims=True)
    esp = fila @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(esp > 0, (T - esp) ** 2 / esp, 0.0))
    k = min(T.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0


filas = []
for etq, pares in [("humanas (Dalpiaz)", pares_dalpiaz()), ("profesor v3", pares_profesor())]:
    datos = [(familia(r), verbo(m)) for r, m in pares]
    datos = [(f, v) for f, v in datos if f and v]
    if not datos:
        print(f"{etq}: sin pares utilizables")
        continue
    C = collections.Counter(datos)
    vfam = collections.Counter(f for f, _ in datos)
    vtot = collections.Counter(v for _, v in datos)
    comunes = [v for v, n in vtot.most_common(25)]
    tab = pd.DataFrame(0, index=sorted(vfam), columns=comunes)
    for (f, v), n in C.items():
        if v in comunes:
            tab.loc[f, v] += n
    V = cramer(tab)
    print(f"\n=== {etq}: {len(pares)} historias con rol, {len(datos)} clasificadas en familia")
    print(f"    V de Cramer sobre rol x verbo (25 verbos mas frecuentes): {V:.3f}")
    print(f"    reparto por familia: {dict(vfam)}")
    N = len(datos)
    for f in sorted(vfam):
        sub = collections.Counter(v for ff, v in datos if ff == f)
        tot_f = sum(sub.values())
        # razon de frecuencia: cuanto mas usa esta familia el verbo que el resto
        razon = []
        for v, n in sub.items():
            if n < 4:
                continue
            p_f = n / tot_f
            p_o = (vtot[v] - n) / max(1, N - tot_f)
            razon.append((p_f / p_o if p_o else 99.0, n, v))
        razon.sort(reverse=True)
        top = ", ".join(f"{v} (x{n}, {r:.1f}v)" for r, n, v in razon[:6])
        print(f"      {f:<11} n={tot_f:<5} caracteristicos: {top}")
        for r, n, v in razon[:6]:
            filas.append(dict(corpus=etq, familia=f, verbo=v, n=n, razon=round(r, 2),
                              cramer_V=round(V, 3)))

pd.DataFrame(filas).to_csv(f"{BASE}/results/82_rol_funcion.csv", index=False)
print("\n-> results/82_rol_funcion.csv")
