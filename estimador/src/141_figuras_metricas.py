"""Figuras de E30-E40: donde falla el modelo y por que no se arregla gratis.

Todas las cifras se leen de los CSV de resultados en tiempo de ejecucion.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
SAL = "reports/figuras_metricas"
os.makedirs(SAL, exist_ok=True)
TINTA, AVISO, AMBAR, GRIS, NEGRO = "#12645d", "#a4442a", "#7d5a17", "#6b7a8a", "#111820"
matplotlib.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce", "axes.labelcolor": "#3d4a58",
    "axes.titlesize": 14, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 12, "axes.grid": True, "grid.color": "#e6ebef",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRIS, "ytick.color": GRIS, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.frameon": False, "legend.fontsize": 11,
    "font.family": "DejaVu Sans", "figure.autolayout": True,
})
NL = chr(10)


def guardar(fig, nombre):
    p = f"{SAL}/{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", p)


def la_trampa():
    """El mismo modelo parece sesgadisimo o insesgado segun como se mire."""
    d = pd.read_csv("results/diagnostico/por_decil.csv")
    s = pd.read_csv("results/diagnostico/sesgo_tramo.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    a1.axhline(1, color=GRIS, lw=1, ls="--")
    a1.plot(range(len(d)), d.sesgo_x, "o-", color=AVISO, lw=2.2, ms=7)
    a1.set_yscale("log")
    a1.set_yticks([.2, .5, 1, 2, 5])
    a1.set_yticklabels(["x0,2", "x0,5", "igual", "x2", "x5"])
    a1.set_xlabel("decil del valor REAL")
    a1.set_title("Visto por decil del valor real:" + NL + "parece que aplasta todo hacia el centro")
    a2.axhline(1, color=GRIS, lw=1, ls="--")
    a2.plot(range(len(s)), s.factor_que_falta, "o-", color=TINTA, lw=2.2, ms=7)
    a2.set_yscale("log")
    a2.set_yticks([.2, .5, 1, 2, 5])
    a2.set_yticklabels(["x0,2", "x0,5", "igual", "x2", "x5"])
    a2.set_ylim(.18, 5.5)
    a2.set_xlabel("tramo de la PREDICCION")
    a2.set_title("Visto por tramo de la prediccion:" + NL + "no hay sesgo que corregir")
    fig.suptitle("La misma prediccion, dos lecturas opuestas", x=.005, ha="left",
                 fontsize=15, weight="600")
    guardar(fig, "1_la_trampa_del_sesgo")


def extremos():
    d = pd.read_csv("results/extremos/extremos.csv")
    fig, a = plt.subplots(figsize=(8.6, 4.4))
    c = [AVISO if v < .7 else TINTA for v in d.AUC]
    b = a.barh(range(len(d)), d.AUC, color=c, height=.62)
    a.axvline(.5, color=GRIS, lw=1.4, ls="--")
    a.axvline(.7, color=AMBAR, lw=1.2, ls=":")
    a.set_yticks(range(len(d)))
    a.set_yticklabels([o.replace("y ", "") for o in d.objetivo])
    a.invert_yaxis()
    a.set_xlim(.45, .82)
    a.set_xlabel("AUC de un clasificador que solo lee el titulo")
    for r, v, p in zip(b, d.AUC, d.prevalencia):
        a.text(v + .005, r.get_y() + r.get_height() / 2, f"{v:.3f}  ({p:.1%} de las tareas)",
               va="center", fontsize=10.5, color=NEGRO)
    a.set_title("Las tareas grandes se ven en el texto. Las microtareas no." + NL +
                "0,5 es no saber nada · 0,7 es empezar a ser util")
    guardar(fig, "2_senal_en_los_extremos")


def frontera():
    """Cada intento, situado entre lo que gana y lo que pierde."""
    fuentes = [("results/distribucional/resumen.csv", "brazo", "distribucional"),
               ("results/forma_residual/resumen.csv", "brazo", "forma del residuo"),
               ("results/pesos_metrica/resumen.csv", "modelo", "pesos por metrica"),
               ("results/apilado_total/resumen.csv", "modelo", "apilado de once")]
    pts = []
    for f, col, fam in fuentes:
        d = pd.read_csv(f)
        d = d[d.bloque == "prueba fija"]
        for _, r in d.iterrows():
            n = str(r[col])
            if n.startswith("    canal") or "constante" in n:
                continue
            pts.append((r.R2log, r.PRED25, n, fam))
    base = [p for p in pts if "actual" in p[2] or "apilado)" in p[2]][0]
    fig, a = plt.subplots(figsize=(9.6, 5.4))
    fams = sorted({p[3] for p in pts})
    cols = dict(zip(fams, [TINTA, AVISO, AMBAR, GRIS]))
    for fam in fams:
        q = [p for p in pts if p[3] == fam]
        a.scatter([p[0] for p in q], [p[1] for p in q], s=54, alpha=.8,
                  color=cols[fam], label=fam, zorder=3)
    a.scatter([base[0]], [base[1]], s=250, marker="*", color=NEGRO, zorder=5,
              label="el modelo actual")
    a.axvline(base[0], color=GRIS, lw=1, ls="--")
    a.axhline(base[1], color=GRIS, lw=1, ls="--")
    a.set_xlabel("R2 sobre log-horas  (capacidad explicativa)")
    a.set_ylabel("PRED(25)  (tasa de acierto)")
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=5)
    a.set_title("Nada cae arriba y a la derecha" + NL +
                "todo lo que sube la tasa de acierto baja la capacidad explicativa")
    guardar(fig, "3_frontera_del_canje")


def canales():
    d = pd.read_csv("results/apilado_total/resumen.csv")
    d = d[(d.bloque == "prueba fija") & d.modelo.str.startswith("    canal")]
    fig, a = plt.subplots(figsize=(9.6, 5.2))
    a.scatter(d.R2log, d.PRED25, s=70, color=TINTA, zorder=3)
    for _, r in d.iterrows():
        n = r.modelo.replace("    canal · ", "")
        es = "analog" in n
        a.annotate(n, (r.R2log, r.PRED25), fontsize=9.5,
                   xytext=(6, 4 if not es else -12), textcoords="offset points",
                   color=AVISO if es else NEGRO, weight="600" if es else "normal")
    ap = pd.read_csv("results/apilado_total/resumen.csv")
    ap = ap[(ap.bloque == "prueba fija") & (ap.modelo == "0 · el modelo actual")].iloc[0]
    a.scatter([ap.R2log], [ap.PRED25], s=250, marker="*", color=NEGRO, zorder=5)
    a.annotate("el apilado de los cuatro", (ap.R2log, ap.PRED25), fontsize=10,
               xytext=(-10, -16), textcoords="offset points", ha="right", weight="600")
    a.set_xlabel("R2 sobre log-horas")
    a.set_ylabel("PRED(25)")
    a.set_title("El apilado empeora la tasa de acierto de sus propios canales" + NL +
                "la analogia por vecinos acierta mas que todo lo demas, y explica menos")
    guardar(fig, "4_que_sabe_cada_canal")


def intervalos():
    d = pd.read_csv("results/conformal/resumen.csv")
    d = d[(d.nivel == "80%") & (d.metodo.str.startswith(("A ", "C ")))]
    ord_b = ["validacion (calibracion)", "prueba fija", "prueba SiP", "externo no Apache"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    x = np.arange(len(ord_b))
    for i, (m, col) in enumerate((("A · anchura constante (conforme)", GRIS),
                                  ("C · conforme cuantilico (adaptativo)", TINTA))):
        v = d[d.metodo == m].set_index("bloque").reindex(ord_b)
        a1.bar(x + (i - .5) * .36, v.cobertura, .34, color=col,
               label=m.split(" · ")[1])
        a2.bar(x + (i - .5) * .36, v.anchura_x, .34, color=col)
    a1.axhline(.8, color=AVISO, lw=1.6, ls="--")
    a1.set_ylim(.6, .88)
    a1.set_xticks(x)
    a1.set_xticklabels([b.split(" (")[0].replace("externo no Apache", "proyectos" + NL + "nuevos")
                        for b in ord_b], fontsize=10)
    a1.set_ylabel("fraccion cubierta")
    a1.legend(loc="lower left")
    a1.set_title("Cobertura de un intervalo que dice ser del 80 %")
    a2.set_xticks(x)
    a2.set_xticklabels([b.split(" (")[0].replace("externo no Apache", "proyectos" + NL + "nuevos")
                        for b in ord_b], fontsize=10)
    a2.set_ylabel("anchura, en factor")
    a2.set_title("Lo que mide ese intervalo")
    fig.suptitle("La garantia conforme se rompe fuera de la validacion", x=.005,
                 ha="left", fontsize=15, weight="600")
    guardar(fig, "5_intervalos")


def panel():
    d = pd.read_csv("results/panel/panel.csv")
    d = d[d.bloque == "prueba fija"]
    ms = [("zeta_%", "exactitud simetrica" + NL + "zeta (%)", False),
          ("MAE_h", "MAE (horas)", False), ("SA", "SA frente al azar", True),
          ("PRED25_%", "PRED(25) (%)", True), ("factor2_%", "factor 2 (%)", True),
          ("R2_log", "R2 en log", True)]
    fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.4))
    for a, (c, etq, mayor) in zip(ax.ravel(), ms):
        v = d.set_index("brazo")[c]
        nb = ["adivinar al azar (referencia)", "modelo en frío", "con estimación del usuario"]
        col = [GRIS, TINTA, AMBAR]
        a.bar(range(3), [v[n] for n in nb], color=col, width=.62)
        a.set_xticks(range(3))
        a.set_xticklabels(["azar", "en frio", "con" + NL + "estimacion"], fontsize=10)
        a.set_title(etq, fontsize=12)
        for i, n in enumerate(nb):
            a.text(i, v[n], f"{v[n]:.3g}", ha="center",
                   va="bottom" if v[n] >= 0 else "top", fontsize=10, color=NEGRO)
        a.margins(y=.22)
    fig.suptitle("El panel que pide el estado del arte, sobre la prueba fija (n = 4.225)",
                 x=.005, ha="left", fontsize=15, weight="600")
    guardar(fig, "6_panel_de_metricas")


def analogia():
    """El barrido entero, para ver que el efecto no depende de la configuracion."""
    d = pd.read_csv("results/analogia/rejilla.csv")
    r = pd.read_csv("results/analogia/resumen.csv")
    base = r[(r.bloque == "prueba fija") &
             (r.brazo == "0 · el apilado actual")].iloc[0].PRED25
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    cols = {"media": GRIS, "mediana": AMBAR, "moda en la rejilla": TINTA}
    for ag, c in cols.items():
        q = d[d.agregador == ag].groupby("k")["prueba fija_PRED25"].mean()
        a1.plot(q.index, q.values, "o-", color=c, lw=2.2, ms=7, label=ag)
    a1.axhline(base, color=AVISO, lw=1.8, ls="--")
    a1.text(50, base, " el apilado", color=AVISO, va="center", fontsize=10.5)
    a1.set_xlabel("k, numero de vecinos")
    a1.set_ylabel("PRED(25) en la prueba fija")
    a1.legend(loc="lower right")
    a1.set_title("Caer sobre un valor que de verdad se observa" + NL +
                 "moda > mediana > media, en toda la rejilla")
    ord_b = ["prueba fija", "prueba SiP", "externo no Apache"]
    x = np.arange(len(ord_b))
    for i, (b, c) in enumerate((("0 · el apilado actual", GRIS),
                                ("2 · encaminado por dispersion de los vecinos", TINTA))):
        v = r[r.brazo == b].set_index("bloque").reindex(ord_b)
        a2.bar(x + (i - .5) * .36, v.PRED25, .34, color=c,
               label="el apilado" if i == 0 else "analogia encaminada")
    a2.set_xticks(x)
    a2.set_xticklabels(["prueba fija", "prueba SiP", "proyectos" + NL + "nuevos"], fontsize=10)
    a2.set_ylabel("PRED(25)")
    a2.set_ylim(.14, .20)
    a2.legend(loc="upper left", ncol=2)
    a2.set_title("Y aguanta en los tres bloques")
    fig.suptitle("Estimacion por analogia: lo unico que sube PRED(25) y factor 2 a la vez",
                 x=.005, ha="left", fontsize=15, weight="600")
    guardar(fig, "7_analogia")




def total_proyecto():
    """La metrica de presentacion: acertar el total de horas del proyecto."""
    r = pd.read_csv("results/mejorar_total/resumen.csv")
    t = pd.read_csv("results/total_proyecto/totales.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    a1.loglog([t.real.min() * .8, t.real.max() * 1.2],
              [t.real.min() * .8, t.real.max() * 1.2], color=GRIS, lw=1.3, ls="--")
    for f, c in ((1.25, AMBAR), (2.0, "#d9e2e8")):
        for k in (f, 1 / f):
            a1.loglog([t.real.min() * .8, t.real.max() * 1.2],
                      [t.real.min() * .8 * k, t.real.max() * 1.2 * k],
                      color=c, lw=1, ls=":")
    a1.scatter(t.real, t.mod_sm, s=26, color=TINTA, alpha=.75, zorder=3)
    a1.set_xlabel("horas reales del proyecto")
    a1.set_ylabel("horas estimadas")
    a1.set_title("114 proyectos, todos predichos como si fuesen nuevos" + NL +
                 "las lineas de puntos son +-25 % y factor 2")
    ks = ["1b · frio, smearing por tramo", "2 · con 1 tarea cerrada",
          "2 · con 3 tareas cerradas", "2 · con 5 tareas cerradas",
          "2 · con 10 tareas cerradas", "2 · con 20 tareas cerradas"]
    v = r.set_index("brazo").reindex(ks)
    x = np.arange(len(ks))
    a2.plot(x, v.PRED25, "o-", color=TINTA, lw=2.4, ms=8, label="dentro del +-25 %")
    a2.plot(x, v.factor2, "o-", color=AMBAR, lw=2.4, ms=8, label="dentro del factor 2")
    cte = r.set_index("brazo").loc["0 · constante (n x mediana)"]
    a2.axhline(cte.PRED25, color=AVISO, lw=1.6, ls="--")
    a2.text(len(ks) - 1, cte.PRED25 + .015, "constante", color=AVISO, ha="right", fontsize=10.5)
    a2.set_xticks(x)
    a2.set_xticklabels(["en frio", "1", "3", "5", "10", "20"])
    a2.set_xlabel("tareas del proyecto ya cerradas")
    a2.set_ylabel("fraccion de proyectos")
    a2.set_ylim(0, .95)
    a2.legend(loc="center right")
    a2.set_title("Y mejora sola en cuanto el equipo cierra tareas")
    fig.suptitle("Total de horas del proyecto: la metrica que se presenta",
                 x=.005, ha="left", fontsize=15, weight="600")
    guardar(fig, "8_total_proyecto")

if __name__ == "__main__":
    for f in (la_trampa, extremos, frontera, canales, intervalos, panel, analogia,
              total_proyecto):
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} FALLO: {type(e).__name__}: {e}")
