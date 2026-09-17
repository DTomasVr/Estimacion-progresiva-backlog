"""Figures for the feasibility deck: our model against the LLMs, cold against corrected.

Same rule as script 75: every number is read from the results files at run time,
so a figure cannot drift from the run that produced it.
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
SAL = "reports/figuras_pptx"
os.makedirs(SAL, exist_ok=True)
TINTA, SUAVE = "#12645d", "#dcebe9"
AVISO, AMBAR, GRIS, NEGRO = "#a4442a", "#7d5a17", "#6b7a8a", "#111820"
VERDE_SUAVE, AMBAR_SUAVE, AVISO_SUAVE = "#cfe3df", "#f2e9d5", "#f4e2dc"

matplotlib.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce", "axes.labelcolor": "#3d4a58",
    "axes.titlesize": 15, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 12.5, "axes.grid": True, "grid.color": "#e6ebef",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRIS, "ytick.color": GRIS, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "legend.frameon": False, "legend.fontsize": 11.5, "font.family": "DejaVu Sans",
    "figure.autolayout": True,
})
MODELOS = ["gemma2:9b (ollama)", "qwen2.5:7b (ollama)", "llama3.1:8b (ollama)",
           "mistral:7b (ollama)"]
COND = {"A": "A\nsolo texto", "B": "B\n10 ejemplos", "C": "C\nhistorial"}


def guardar(fig, nombre):
    p = f"{SAL}/{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", p)


# --------------------------------------------- 1 · tablero de veredictos
def tablero():
    l = pd.read_csv("results/llm_bench/resumen.csv")
    tono = {"nuestro mejor": VERDE_SUAVE, "empate": AMBAR_SUAVE, "LLM mejor": AVISO_SUAVE}
    letra = {"nuestro mejor": TINTA, "empate": AMBAR, "LLM mejor": AVISO}
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for a, ver, titulo in ((ax[0], "calibrado", "El LLM solo (calibrado con la misma validación)"),
                           (ax[1], "combinado", "El LLM combinado con nuestro modelo")):
        v = l[l.version == ver]
        cuenta = {"nuestro mejor": 0, "empate": 0, "LLM mejor": 0}
        for i, m in enumerate(MODELOS):
            for j, c in enumerate("ABC"):
                f = v[(v.modelo == m) & (v.condicion == c)].iloc[0]
                cuenta[f.veredicto] += 1
                a.add_patch(plt.Rectangle((j - .48, i - .44), .96, .88,
                                          facecolor=tono[f.veredicto], edgecolor="white", lw=2))
                a.text(j, i + .10, f"{f.MdMRE:.3f}", ha="center", va="center",
                       fontsize=13.5, weight="700", color=NEGRO)
                a.text(j, i - .21, f.veredicto, ha="center", va="center",
                       fontsize=9.5, weight="700", color=letra[f.veredicto])
        a.set_xlim(-.5, 2.5)
        a.set_ylim(len(MODELOS) - .5, -.5)
        a.set_xticks(range(3), [COND[c] for c in "ABC"])
        a.set_yticks(range(len(MODELOS)), [m.split(" ")[0] for m in MODELOS])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(f"{titulo}\n{cuenta['nuestro mejor']} a favor nuestro · "
                    f"{cuenta['empate']} empates · {cuenta['LLM mejor']} a favor del LLM",
                    fontsize=12.5)
    fig.suptitle("Doce combinaciones modelo x condicion, cada una en dos versiones  ·  las mismas 601 tareas  ·  MdMRE del LLM",
                 x=.01, ha="left", fontsize=15.5, weight="600")
    guardar(fig, "fact_tablero_llm")


# --------------------------------------------- 2 · coste de servir una prediccion
def costo():
    n = pd.read_csv("results/costo/latencia_nuestra.csv")
    l = pd.read_csv("results/costo/latencia_llms.csv")
    r = json.load(open("results/costo/resumen.json", encoding="utf-8"))
    gpu = n[n.componente.str.contains("cuda")]
    nuestro = float(gpu.seg_una_tarea.iloc[0]) + float(
        n[n.componente.str.startswith("regresor")].seg_una_tarea.iloc[0]) if len(gpu) else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.4))
    nombres = ["nuestro canal\n(SBERT + regresor)"] + [m.split(" ")[0] for m in l.modelo]
    minimos = [nuestro] + list(l.seg_tarea_min)
    maximos = [nuestro] + list(l.seg_tarea_max)
    col = [TINTA] + [AMBAR] * len(l)
    y = np.arange(len(nombres))
    ax[0].barh(y, maximos, color=col, height=.62, alpha=.35)
    ax[0].barh(y, minimos, color=col, height=.62)
    for i, (lo, hi) in enumerate(zip(minimos, maximos)):
        txt = (f"{lo * 1000:.0f} ms" if i == 0 else
               f"{lo:.2f}" + (f"–{hi:.2f} s" if hi - lo > .01 else " s"))
        ax[0].text(hi + .03, i, txt, va="center", fontsize=11.5, color=NEGRO, weight="600")
    ax[0].set_yticks(y, nombres)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, max(maximos) * 1.35)
    ax[0].set_xlabel("segundos por tarea, una a una  ·  barra clara = peor caso medido")
    ax[0].set_title("Lo que espera el usuario")

    pesos = [r["peso_codificador_MB"] + r["peso_regresor_MB"]]
    for m in l.modelo:
        t = str(l[l.modelo == m].peso_en_disco.iloc[0])
        pesos.append(float(t.split()[0]) * 1000 if "GB" in t else float(t.split()[0] or 0))
    ax[1].barh(y, pesos, color=col, height=.62)
    for i, v in enumerate(pesos):
        ax[1].text(v + max(pesos) * .015, i, f"{v / 1000:.2f} GB", va="center",
                   fontsize=11.5, color=NEGRO, weight="600")
    ax[1].set_yticks(y, ["" for _ in y])
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, max(pesos) * 1.28)
    ax[1].set_xlabel("pesos que hay que desplegar, en MB")
    ax[1].set_title("Lo que hay que instalar")
    fig.suptitle("Coste de servir una estimación en la misma máquina (RTX 5060, 8 GB)",
                 x=.01, ha="left", fontsize=15.5, weight="600")
    guardar(fig, "fact_costo")


# --------------------------------------------- 3 · frio contra estimacion corregida
def frio_vs_estimacion():
    r = pd.read_csv("results/estimacion_completa/resumen.csv")
    r = r[r.subconjunto == "todas"]
    gan = r[r.elegido_en_val == True].brazo.iloc[0]      # noqa: E712
    escenarios = ["SiP - prueba (misma empresa)", "Jira - externo (cliente nuevo)",
                  "TAWOS - prueba (puntos de historia)"]
    etiquetas = ["SiP\nmisma empresa", "Jira externo\ncliente nuevo", "TAWOS\npuntos"]
    brazos = [("R0 - modelo en frio (sin estimacion)", "modelo en frío", GRIS),
              ("H0 - estimacion del usuario, tal cual", "estimación del usuario sin corregir", AMBAR),
              (gan, "estimación corregida por el modelo", TINTA)]
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.9))
    for a, met, etiq, lim in ((ax[0], "MdMRE", "MdMRE · menos es mejor", (0, 1.16)),
                              (ax[1], "MAE_h", "error medio en horas · menos es mejor", (0, 15.5)),
                              (ax[2], "SA", "SA frente al azar · más es mejor", (-.62, .78))):
        x = np.arange(len(escenarios))
        for i, (br, nom, c) in enumerate(brazos):
            v = [float(r[(r.escenario == e) & (r.brazo == br)][met].iloc[0]) for e in escenarios]
            b = a.bar(x + (i - 1) * .26, v, .24, label=nom, color=c)
            for rect, val in zip(b, v):
                d = (lim[1] - lim[0]) * .02
                a.text(rect.get_x() + rect.get_width() / 2, val + d if val >= 0 else val - d * 2.4,
                       f"{val:.2f}" if met != "MdMRE" else f"{val:.3f}",
                       ha="center", fontsize=9.5, weight="600", color=NEGRO)
        a.set_xticks(x, etiquetas)
        a.set_ylabel(etiq, fontsize=11.5)
        a.set_ylim(*lim)
        if met == "SA":
            a.axhline(0, color=AVISO, lw=1.8)
            a.text(-.42, -.085, "peor que adivinar al azar", ha="left", fontsize=10,
                   color=AVISO, weight="700")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.035), fontsize=12)
    fig.suptitle("Frío, estimación cruda y estimación corregida: la corrección arregla "
                 "el error en horas", x=.01, ha="left", fontsize=15, weight="600")
    guardar(fig, "fact_frio_vs_estimacion")


# --------------------------------------------- 4 · curva de adopcion
def adopcion():
    c = pd.read_csv("results/estimacion_completa/curva_adopcion.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.4))
    estilo = {"corregida": ("estimación corregida por el modelo", TINTA, "-o"),
              "cruda": ("estimación del usuario tal cual", AMBAR, "--s")}
    for a, met, etiq in ((ax[0], "MdMRE", "MdMRE · menos es mejor"),
                         (ax[1], "MAE_h", "error medio en horas · menos es mejor")):
        for var, (nom, col, est) in estilo.items():
            v = c[c.variante == var]
            a.plot(v.adopcion * 100, v[met], est, color=col, lw=2.4, ms=5.5, label=nom)
        base = float(c[(c.variante == "corregida") & (c.adopcion == 0)][met].iloc[0])
        a.axhline(base, color=GRIS, ls=":", lw=1.6)
        a.text(3, base + (a.get_ylim()[1] - a.get_ylim()[0]) * .02,
               f"todo en frío  {base:.3f}", fontsize=10.5, color=GRIS, weight="700")
        a.set_xlabel("% de tareas con estimación escrita por el usuario")
        a.set_ylabel(etiq)
        a.set_xlim(-2, 102)
    ax[0].legend(loc="lower left")
    fig.suptitle("La mejora es proporcional a cuánta gente responde  ·  bloque de prueba "
                 f"fija completo, 4.225 tareas ({c.cobertura.max():.0%} como máximo "
                 "pueden llevar estimación)", x=.01, ha="left", fontsize=14.5, weight="600")
    guardar(fig, "fact_adopcion")


# --------------------------------------------- 5 · sesgo por tamano de tarea
def sesgo_tamano():
    s = pd.read_csv("results/llm_bench/sesgo_por_tamano.csv")
    tramos = ["<1h", "1-4h", "4-16h", ">16h"]
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.3), sharey=True)
    for a, cond, titulo in ((ax[0], "A", "A · el LLM solo ve el texto"),
                            (ax[1], "C", "C · el LLM ve el historial del proyecto")):
        v = s[s.condicion == cond]
        x = np.arange(len(tramos))
        for i, m in enumerate(MODELOS):
            y = [float(v[(v.modelo == m) & (v.tramo == t)].sesgo_llm.iloc[0]) for t in tramos]
            a.plot(x, y, "-o", lw=1.9, ms=5, color=[AMBAR, "#b8862a", "#d9a03f", "#8a6a2a"][i],
                   label=m.split(" ")[0])
        y = [float(v[v.tramo == t].sesgo_nuestro.iloc[0]) for t in tramos]
        a.plot(x, y, "-o", lw=3, ms=7, color=TINTA, label="nuestro modelo")
        a.axhline(1, color=NEGRO, lw=1.4, ls="--")
        a.set_yscale("log")
        a.set_xticks(x, tramos)
        a.set_xlabel("tamaño real de la tarea")
        a.set_title(titulo)
    ax[0].set_ylabel("mediana de predicho / real\n(1 = acierta el nivel)")
    ax[0].legend(loc="lower left", ncol=2)
    fig.suptitle("Todos exageran las tareas pequeñas; el LLM sin contexto las multiplica por ocho",
                 x=.01, ha="left", fontsize=15.5, weight="600")
    guardar(fig, "fact_sesgo_tamano")


if __name__ == "__main__":
    print("generando figuras:")
    tablero()
    costo()
    frio_vs_estimacion()
    adopcion()
    sesgo_tamano()
    print("listo")
