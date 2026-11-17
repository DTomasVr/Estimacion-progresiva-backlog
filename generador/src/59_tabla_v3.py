"""Junta los resultados de la tanda v3 en una tabla y la escribe en
entrega/ROLES_Y_COHERENCIA.md, seccion 5.

Lee lo que dejaron las fases de src/run_v3.sh, sin volver a generar nada:
  logs/52_{base,v3a,v3r}.log     roles, relleno, eco, anclaje, atribucion
  results/57_arista_{...}.csv    F1 de nodo y de arista, mala atribucion
  logs/49_{v3a,v3r}.log          328 historias humanas, pareado contra produccion
  results/58_ablacion.csv        que aportan los filtros

  python src/59_tabla_v3.py
"""
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAR = [("base", "produccion (30 sectores)"),
       ("v3a", "A: filtrar"),
       ("v3r", "B: ficha de rol"),
       ("v3b", "A sin despublicar")]

PATRONES = {
    "un_solo_rol": r"un solo rol en el backlog\s*:\s*([\d.]+)",
    "roles_media": r"media ([\d.]+) roles",
    "cuota_dom": r"cuota del rol dominante\s*:\s*([\d.]+)",
    "relleno": r"relleno de plantilla\s*:\s*([\d.]+)",
    "conc": r"concentracion top5 arranques\s*:\s*([\d.]+)",
    "eco": r"eco \(repite un sustantivo\)\s*:\s*([\d.]+)",
    "anclada": r"ancladas a SU historia\s*:\s*([\d.]+)",
    "atrib": r"acierto \(tarea -> su historia\)\s*:\s*([\d.]+)",
}


def de_log_52(etq):
    # La concentracion de arranques se anadio despues de medir base, v3a y v3r,
    # asi que esas tres se volvieron a medir en logs/52_<etq>_conc.log. Se leen
    # los dos ficheros y el segundo pisa al primero donde tenga dato.
    out = {}
    for p in (f"{BASE}/logs/52_{etq}.log", f"{BASE}/logs/52_{etq}_conc.log"):
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8", errors="replace").read()
        for k, r in PATRONES.items():
            m = re.search(r, t)
            if m:
                out[k] = float(m.group(1))
    return out


def de_csv_57(etq):
    p = f"{BASE}/results/57_arista_{etq}.csv"
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    return {k: float(d[k].mean()) for k in ["hist_F1", "nodo_F1", "arista_F1", "mala_atrib"]
            if k in d.columns}


def de_log_49(etq):
    p = f"{BASE}/logs/49_{etq}.log"
    if not os.path.exists(p):
        return {}
    t = open(p, encoding="utf-8", errors="replace").read()
    out = {}
    for etiqueta in ("viejo", "nuevo"):
        m = re.search(rf"##### {etiqueta}: sF1 conjunto ([\d.]+)", t)
        if m:
            out[f"sF1_{etiqueta}"] = float(m.group(1))
    m = re.search(r"nuevo - viejo sobre historias HUMANAS: ([+-][\d.]+) "
                  r"IC95 \[([+-][\d.]+), ([+-][\d.]+)\]", t)
    if m:
        out["dif"], out["lo"], out["hi"] = (float(m.group(i)) for i in (1, 2, 3))
    m = re.search(r"gana el nuevo en (\d+) de (\d+)", t)
    if m:
        out["gana"] = f"{m.group(1)}/{m.group(2)}"
    return out


# De los 6 proyectos del test humano, DOS tienen ahora un sector hecho a medida
# entre los 10 nuevos de la tanda v3: 'school' con el sector colegio y
# 'supermarket' con el sector supermercado. No hay fuga de textos (las
# descripciones sinteticas se escriben desde la taxonomia, no desde las historias
# del test), pero si hay solapamiento de DOMINIO, asi que una mejora en esos dos
# puede venir de haberles construido datos a medida. Los otros cuatro ya estaban
# cubiertos al mismo nivel que antes de v3 (restaurantes, eventos, fitness, rrhh
# existian desde la tanda v1), asi que en ellos la comparacion es limpia.
SOLAPADOS = {"school", "supermarket"}


def humanas_limpias(etq):
    """diferencia contra produccion solo en los 4 proyectos sin sector a medida"""
    p = f"{BASE}/results/49_real_{etq}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if "variante" not in d.columns or d.variante.nunique() < 2:
        return None
    t = d.pivot_table(index="proyecto", columns="variante", values="sF1")
    if not {"viejo", "nuevo"} <= set(t.columns):
        return None
    lim = t[~t.index.isin(SOLAPADOS)]
    return dict(n=len(lim), dif=float((lim["nuevo"] - lim["viejo"]).mean()),
                gana=int((lim["nuevo"] > lim["viejo"]).sum()),
                n_sol=len(t) - len(lim),
                dif_sol=float((t[t.index.isin(SOLAPADOS)]["nuevo"]
                               - t[t.index.isin(SOLAPADOS)]["viejo"]).mean())
                if len(t) > len(lim) else None)


def de_csv_57_espejo(etq):
    p = f"{BASE}/results/57_arista_{etq}_refv3.csv"
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    return {f"{k}_v3ref": float(d[k].mean()) for k in ["nodo_F1", "arista_F1"]
            if k in d.columns}


def pct(x):
    return "—" if x is None else f"{x:.1f} %"


def num(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


D = {e: {**de_log_52(e), **de_csv_57(e), **de_csv_57_espejo(e), **de_log_49(e)}
     for e, _ in VAR}
G = lambda e, k: D[e].get(k)          # noqa: E731

filas = [
    ("**roles**", None),
    ("un solo rol en todo el backlog", *[pct(G(e, "un_solo_rol")) for e, _ in VAR]),
    ("roles distintos por backlog", *[num(G(e, "roles_media"), 2) for e, _ in VAR]),
    ("cuota del rol dominante", *[pct(G(e, "cuota_dom")) for e, _ in VAR]),
    ("**tareas**", None),
    ("relleno de plantilla (lista fija, ciega)", *[pct(G(e, "relleno")) for e, _ in VAR]),
    ("**concentracion top5 arranques** (humanos 2,6 %)",
     *[pct(G(e, "conc")) for e, _ in VAR]),
    ("eco", *[pct(G(e, "eco")) for e, _ in VAR]),
    ("ancladas a su historia", *[pct(G(e, "anclada")) for e, _ in VAR]),
    ("**estructura**", None),
    ("atribucion tarea->historia (sin referencia)", *[pct(G(e, "atrib")) for e, _ in VAR]),
    ("F1 de nodo (tareas, contra referencia)", *[num(G(e, "nodo_F1")) for e, _ in VAR]),
    ("**F1 de ARISTA**", *[num(G(e, "arista_F1")) for e, _ in VAR]),
    ("F1 de nodo, referencias v3", *[num(G(e, "nodo_F1_v3ref")) for e, _ in VAR]),
    ("**F1 de ARISTA, referencias v3**", *[num(G(e, "arista_F1_v3ref")) for e, _ in VAR]),
    ("**328 historias humanas**", None),
    ("sF1", *[num(G(e, "sF1_nuevo") or G(e, "sF1_viejo")) for e, _ in VAR]),
    ("diferencia contra produccion", "—",
     *[f"{num(G(e,'dif'))} IC95 [{num(G(e,'lo'))}, {num(G(e,'hi'))}]"
       if G(e, "dif") is not None else "—" for e, _ in VAR[1:]]),
    ("gana en", "—", *[G(e, "gana") or "—" for e, _ in VAR[1:]]),
]

LIM = {e: humanas_limpias(e) for e, _ in VAR[1:]}
filas.append(("**sin los 2 proyectos con sector a medida**", None))
filas.append(("diferencia en los 4 limpios", "—",
              *[f"{LIM[e]['dif']:+.4f} (gana {LIM[e]['gana']}/{LIM[e]['n']})"
                if LIM.get(e) else "—" for e, _ in VAR[1:]]))
filas.append(("diferencia en school y supermarket", "—",
              *[f"{LIM[e]['dif_sol']:+.4f}"
                if LIM.get(e) and LIM[e]["dif_sol"] is not None else "—"
                for e, _ in VAR[1:]]))

anchos = ["medida"] + [n for _, n in VAR]
lineas = ["| " + " | ".join(anchos) + " |", "|" + "---|" * (len(VAR) + 1)]
for f in filas:
    if f[1] is None:
        lineas.append("| " + f[0] + " |" + " |" * len(VAR))
    else:
        lineas.append("| " + " | ".join(str(x) for x in f) + " |")
tabla = "\n".join(lineas)

abl = ""
p = f"{BASE}/results/58_ablacion.csv"
if os.path.exists(p):
    a = pd.read_csv(p)
    abl = "\n\n### Qué aportan los filtros de inferencia (misma población)\n\n"
    abl += "| caso | rol válido | tareas degeneradas | tareas por historia |\n|---|---|---|---|\n"
    for _, r in a.iterrows():
        abl += (f"| {r['caso']} | {100*r['rol_valido']:.1f} % | "
                f"{100*r['degeneradas']:.1f} % | {r['tareas_por_hist']:.1f} |\n")

texto = ("## 5. Resultados\n\nMedido con un solo generador abajo en las tres columnas, para que la "
         "comparación sea pareada.\n\n" + tabla + abl +
         "\n\nLas referencias del F1 de nodo y de arista las escribió el profesor **v1**, que usaba "
         "un solo rol por backlog. Un modelo que reparte bien los actores puede por eso parecer peor "
         "en el nivel de historia sin serlo; lo que no sufre ese sesgo es el F1 de arista, que mide "
         "estructura, y las 328 historias humanas.\n\n"
         "**Solapamiento declarado.** De los 6 proyectos del test humano, dos —`school` y "
         "`supermarket`— tienen ahora un sector hecho a medida entre los 10 nuevos de v3 "
         "(`colegio` y `supermercado`). No hay fuga de textos: las descripciones sintéticas se "
         "escriben desde la taxonomía, no desde las historias del test. Pero sí hay solapamiento "
         "de dominio, así que la cifra que hay que mirar es la de los **4 proyectos limpios** "
         "(`fish_chips`, `ticket`, `grocery`, `sports`), cuyos dominios ya estaban cubiertos al "
         "mismo nivel desde la tanda v1. Las dos filas están separadas arriba a propósito.\n")

doc = f"{BASE}/entrega/ROLES_Y_COHERENCIA.md"
s = open(doc, encoding="utf-8").read()
i = s.index("## 5. Resultados")
open(doc, "w", encoding="utf-8").write(s[:i] + texto)
print(tabla + abl)
print(f"\n-> escrito en entrega/ROLES_Y_COHERENCIA.md")
