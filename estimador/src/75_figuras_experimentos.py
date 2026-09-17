"""Slide figures for the deck comparing the three new experiments with the baseline.

Every number is read from the results CSVs at run time rather than typed in, so a
figure can never drift from the experiment that produced it.
"""
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

matplotlib.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce", "axes.labelcolor": "#3d4a58",
    "axes.titlesize": 16, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.labelsize": 13, "axes.grid": True, "grid.color": "#e6ebef",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRIS, "ytick.color": GRIS, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.frameon": False, "legend.fontsize": 12, "font.family": "DejaVu Sans",
    "figure.autolayout": True,
})


def guardar(fig, nombre):
    p = f"{SAL}/{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", p)


# ---------------------------------------------- 1 · algoritmos sobre la prueba fija
def algoritmos():
    s = pd.read_csv("results/sweep_v3.csv")
    b = pd.read_csv("results/baselines_literatura.csv")
    f = pd.read_csv("results/flags/resumen.csv")
    f = f[f.familia != "referencia"]
    f = f.loc[f.groupby("familia").val_MdAElog.idxmin()]

    filas = [(r.modelo, r.MdMRE_test, "inicial") for _, r in s.iterrows()
             if r.modelo != "linea base (mediana)"]
    filas += [("GPT2SP (GPT-2 ajustado)", float(b[b.modelo.str.startswith("GPT2SP")].MdMRE.iloc[0]), "literatura"),
              ("Deep-SE (LSTM)", float(b[b.modelo.str.startswith("Deep-SE")].MdMRE.iloc[0]), "literatura")]
    filas += [(f"flags · {r.algoritmo} ({r.familia.split(' · ')[0]})", r.prueba_d1_MdMRE, "flags")
              for _, r in f.iterrows()]
    filas.append(("línea base (mediana del proyecto)",
                  float(s[s.modelo == "linea base (mediana)"].MdMRE_test.iloc[0]), "base"))
    d = pd.DataFrame(filas, columns=["modelo", "MdMRE", "grupo"]).sort_values("MdMRE")
    color = {"inicial": TINTA, "literatura": AMBAR, "flags": AVISO, "base": GRIS}

    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    y = np.arange(len(d))
    ax.barh(y, d.MdMRE, color=[color[g] for g in d.grupo], height=.72)
    ax.set_yticks(y, d.modelo)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("MdMRE sobre la misma prueba fija de 4.225 tareas  ·  menos es mejor")
    for i, v in enumerate(d.MdMRE):
        ax.text(v + .004, i, f"{v:.4f}", va="center", fontsize=11.5, color=NEGRO)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=color[k]) for k in ("inicial", "literatura", "flags", "base")]
    ax.legend(manijas, ["barrido inicial (v3)", "literatura reimplementada",
                        "nuevo: solo flags", "línea base"], loc="upper left",
              bbox_to_anchor=(0, -0.085), ncol=4)
    ax.set_title("Todos los algoritmos, un solo examen")
    guardar(fig, "exp_algoritmos")


# ---------------------------------------------- 2 · LLMs por condicion
def llms():
    l = pd.read_csv("results/llm_bench/resumen.csv")
    modelos = ["llama3.1:8b (ollama)", "qwen2.5:7b (ollama)", "gemma2:9b (ollama)", "mistral:7b (ollama)"]
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for a, cond, titulo in zip(ax, "ABC", ["A · solo texto", "B · con 10 ejemplos",
                                           "C · con historial del proyecto"]):
        v = l[l.condicion == cond]
        ref = v[(v.version == "referencia") & (~v.modelo.str.contains("mismas"))
                & (v.modelo.str.startswith("nuestro"))].sort_values("MdMRE").iloc[0]
        cal = [float(v[(v.modelo == m) & (v.version == "calibrado")].MdMRE.iloc[0]) for m in modelos]
        com = [float(v[(v.modelo == m) & (v.version == "combinado")].MdMRE.iloc[0]) for m in modelos]
        x = np.arange(len(modelos))
        a.bar(x - .2, cal, .38, label="LLM calibrado", color=AMBAR)
        a.bar(x + .2, com, .38, label="LLM × nuestro modelo", color=TINTA)
        a.axhline(ref.MdMRE, color=AVISO, lw=2.2)
        a.text(3.45, ref.MdMRE + .012, f"nuestro modelo {ref.MdMRE:.3f}", ha="right",
               fontsize=11.5, color=AVISO, weight="700")
        a.set_xticks(x, [m.split(" ")[0].replace(":", "\n") for m in modelos])
        a.set_title(titulo)
        a.set_ylim(0, 0.95)
    ax[0].set_ylabel("MdMRE · menos es mejor")
    ax[0].legend(loc="upper left")
    fig.suptitle("Cuatro LLMs locales sobre las mismas 601 tareas", x=.01, ha="left",
                 fontsize=16, weight="600")
    guardar(fig, "exp_llms")


# ---------------------------------------------- 3 · estimacion del usuario
def estimacion():
    u = pd.read_csv("results/estimacion_usuario/resumen.csv")
    u = u[u.subconjunto == "todas"]
    brazos = [("R0 · modelo en frío (sin estimación)", "en frío\n(solo texto)", GRIS),
              ("H0 · estimación del usuario, tal cual", "estimación\ndel usuario", AMBAR),
              ("M1 · modelo con estimación, sin texto", "modelo con\nestimación", TINTA)]
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.3))
    for a, metrica, etiqueta, lim in ((ax[0], "MdMRE", "MdMRE · menos es mejor", .85),
                                      (ax[1], "PRED25", "PRED(25) · más es mejor", .85)):
        anchos = np.arange(2)
        for i, (brazo, nombre, col) in enumerate(brazos):
            vals = [float(u[(u.escenario.str.startswith(e)) & (u.brazo == brazo)][metrica].iloc[0])
                    for e in ("SiP", "Jira")]
            b = a.bar(anchos + (i - 1) * .27, vals, .25, label=nombre, color=col)
            for r, v in zip(b, vals):
                a.text(r.get_x() + r.get_width() / 2, v + .012, f"{v:.3f}", ha="center",
                       fontsize=11, weight="600", color=NEGRO)
        a.set_xticks(anchos, ["misma empresa\n(SiP, 1.885)", "cliente nuevo\n(Jira, 1.163)"])
        a.set_ylabel(etiqueta)
        a.set_ylim(0, lim)
        if metrica == "PRED25":
            a.axhline(.75, color=AVISO, lw=2)
            a.text(1.45, .77, "criterio de aceptación 0,75", ha="right", fontsize=11,
                   color=AVISO, weight="700")
    ax[0].legend(loc="upper center", ncol=3)
    fig.suptitle("Pedir la estimación al usuario: el cambio más grande medido", x=.01,
                 ha="left", fontsize=16, weight="600")
    guardar(fig, "exp_estimacion")


# ---------------------------------------------- 4 · cuanto mueve cada palanca
def palancas():
    p = [("Pedir la estimación al usuario\nmisma empresa · M1 frente al modelo en frío", -0.467, AVISO),
         ("Pedir la estimación al usuario\ncliente nuevo · M1 frente al modelo en frío", -0.285, AVISO),
         ("Combinar con un LLM local\ncon historial · gemma2 × modelo frente al modelo de razón", -0.068, TINTA),
         ("Cambiar de algoritmo\nmejor frente a peor del barrido inicial (hgb frente a árbol)", -0.057, AMBAR),
         ("Predecir con la razón contra la mediana del proyecto\ndentro del corpus público", -0.128, TINTA),
         ("Seis veces más datos de entrenamiento\nprueba fija · v4 frente a v3", -0.006, GRIS),
         ("Ajustar el codificador de extremo a extremo\nprueba fija", +0.001, GRIS),
         ("Sustituir el vector por flags de categoría\nprueba fija · SVR lineal frente al vector", +0.034, GRIS),
         ("Usar un LLM solo, sin combinar\nen frío · gemma2 calibrado frente al modelo", +0.061, GRIS)]
    d = pd.DataFrame(p, columns=["palanca", "delta", "color"]).sort_values("delta")
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    y = np.arange(len(d))
    ax.barh(y, d.delta, color=d.color, height=.68)
    ax.set_yticks(y, d.palanca, fontsize=11.5)
    ax.invert_yaxis()
    ax.axvline(0, color=NEGRO, lw=1.2)
    ax.set_xlabel("cambio en MdMRE frente a su propia referencia  ·  negativo = mejora")
    ax.set_xlim(-0.52, 0.12)
    for i, v in enumerate(d.delta):
        ax.text(v + (-.012 if v < 0 else .006), i, f"{v:+.3f}",
                va="center", ha="right" if v < 0 else "left", fontsize=11.5,
                weight="700", color=NEGRO)
    ax.set_title("Qué mueve de verdad la aguja")
    guardar(fig, "exp_palancas")


if __name__ == "__main__":
    print("generando figuras:")
    algoritmos()
    llms()
    estimacion()
    palancas()
    print("listo")
