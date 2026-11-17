"""Figuras de E30-E40: donde falla el modelo y por que no se arregla gratis.

Todas las cifras se leen de los CSV de resultados en tiempo de ejecucion.
"""
import os
import re

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
    r = pd.read_csv("results/curva_final/resumen.csv")
    t = pd.read_csv("results/curva_final/totales.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    a1.loglog([t.real.min() * .8, t.real.max() * 1.2],
              [t.real.min() * .8, t.real.max() * 1.2], color=GRIS, lw=1.3, ls="--")
    for f, c in ((1.25, AMBAR), (2.0, "#d9e2e8")):
        for k in (f, 1 / f):
            a1.loglog([t.real.min() * .8, t.real.max() * 1.2],
                      [t.real.min() * .8 * k, t.real.max() * 1.2 * k],
                      color=c, lw=1, ls=":")
    a1.scatter(t.real, t.frio, s=26, color=TINTA, alpha=.75, zorder=3)
    a1.set_xlabel("horas reales del proyecto")
    a1.set_ylabel("horas estimadas")
    a1.set_title("114 proyectos, todos predichos como si fuesen nuevos" + NL +
                 "las lineas de puntos son +-25 % y factor 2")
    # Las etiquetas se leen del CSV. Antes estaban escritas a mano con un sufijo
    # "(cociente de sumas)" que el CSV no tiene, asi que el reindex devolvia NaN y
    # la curva no se dibujaba: solo salia el punto de "en frio".
    br = r.set_index("brazo")
    cerradas = sorted([b for b in br.index if b.startswith("2 · con ")],
                      key=lambda b: int(re.search(r"con (\d+)", b).group(1)))
    ks = ["1 · en frio"] + cerradas
    v = br.reindex(ks)
    assert not v.PRED25.isna().any(), f"faltan brazos en el CSV: {ks}"
    etiq = ["en frio"] + [re.search(r"con (\d+)", b).group(1) for b in cerradas]
    x = np.arange(len(ks))
    a2.plot(x, v.PRED25, "o-", color=TINTA, lw=2.4, ms=8, label="dentro del +-25 %")
    a2.plot(x, v.factor2, "o-", color=AMBAR, lw=2.4, ms=8, label="dentro del factor 2")
    cte = r.set_index("brazo").loc["0 · constante (n x mediana)"]
    a2.axhline(cte.PRED25, color=AVISO, lw=1.6, ls="--")
    a2.text(len(ks) - 1, cte.PRED25 + .015, "constante", color=AVISO, ha="right", fontsize=10.5)
    a2.set_xticks(x)
    a2.set_xticklabels(etiq)
    a2.set_xlabel("tareas del proyecto ya cerradas")
    a2.set_ylabel("fraccion de proyectos")
    a2.set_ylim(0, .95)
    a2.legend(loc="upper left")
    a2.set_title("Y mejora sola en cuanto el equipo cierra tareas")
    fig.suptitle("Total de horas del proyecto: la metrica que se presenta",
                 x=.005, ha="left", fontsize=15, weight="600")
    guardar(fig, "8_total_proyecto")



def compresion_escala():
    """Lo que el modelo sabe y lo que no: el numero de tareas y el tamano tipico."""
    import numpy as np
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    t = d.groupby("p").size()
    d = d[d.p.isin(t[t >= 8].index)]
    g = d.groupby("p")
    tr = g.real.median().to_numpy()
    tp = g.est.median().to_numpy()
    n = g.size().to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    lim = [min(tr.min(), tp.min()) * .7, max(tr.max(), tp.max()) * 1.4]
    a1.loglog(lim, lim, color=GRIS, lw=1.3, ls="--")
    a1.scatter(tr, tp, s=30, color=TINTA, alpha=.75, zorder=3)
    a1.set_xlim(*lim)
    a1.set_ylim(*lim)
    a1.set_xlabel("tamano tipico REAL de la tarea (h)")
    a1.set_ylabel("tamano tipico predicho (h)")
    a1.set_title("No distingue un equipo de tickets de media" + NL +
                 "hora de uno de tareas de ocho")
    b = np.polyfit(np.log(tr), np.log(tp), 1)[0]
    a1.text(.04, .93, f"la nube es horizontal: sd predicha {np.log(tp).std():.2f}" + NL +
            f"frente a sd real {np.log(tr).std():.2f}", transform=a1.transAxes,
            fontsize=11, color=AVISO, va="top")
    d2 = pd.read_csv("results/donde_falla/proyectos.csv")
    d2["tramo"] = pd.cut(d2.h_por_tarea, [0, .5, 1, 2, 4, 1e9],
                         labels=["<0,5", "0,5-1", "1-2", "2-4", ">4"])
    r = d2.groupby("tramo", observed=True).apply(
        lambda q: pd.Series(dict(sesgo=q.razon.median(), n=len(q))))
    a2.axhline(1, color=GRIS, lw=1.3, ls="--")
    b2 = a2.bar(range(len(r)), r.sesgo, color=[AVISO if v > 1.6 or v < .7 else TINTA
                                               for v in r.sesgo], width=.62)
    a2.set_yscale("log")
    a2.set_yticks([.5, 1, 2, 5])
    a2.set_yticklabels(["x0,5", "igual", "x2", "x5"])
    a2.set_yticks([], minor=True)
    a2.set_ylim(.45, 7.5)
    a2.set_xticks(range(len(r)))
    a2.set_xticklabels([f"{i}" + NL + f"({int(v)} proy.)" for i, v in
                        zip(r.index.astype(str), r.n)], fontsize=10)
    a2.set_xlabel("mediana de horas por tarea del proyecto")
    a2.set_title("Y por eso se pasa con los de tareas cortas" + NL +
                 "y se queda corto con los de tareas largas")
    for rr, v in zip(b2, r.sesgo):
        a2.text(rr.get_x() + rr.get_width() / 2, v, f"x{v:.2f}", ha="center",
                va="bottom" if v >= 1 else "top", fontsize=10.5, color=NEGRO)
    fig.suptitle("Lo que el texto no dice: la escala del proyecto", x=.005,
                 ha="left", fontsize=15, weight="600")
    guardar(fig, "9_compresion_escala")



def anclas_escala():
    """Lo que de verdad arregla la escala: tareas cerradas."""
    import numpy as np
    r = pd.read_csv("results/anclas_escala/resumen.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.9))
    xc = np.arange(len(r))
    a1.axhline(0.791, color=AVISO, lw=1.6, ls="--")
    a1.text(xc[-1], .791, " dispersion real ", color=AVISO, va="bottom", ha="right",
            fontsize=10.5)
    a1.plot(xc, r.sd, "o-", color=TINTA, lw=2.4, ms=8, label="dispersion de la escala")
    a1.plot(xc, r.spearman, "o-", color=AMBAR, lw=2.4, ms=8,
            label="calidad del orden (Spearman)")
    a1.set_xticks(xc)
    a1.set_xticklabels(["en frio" if k == 0 else str(int(k)) for k in r.k])
    a1.set_xlabel("tareas del proyecto ya cerradas")
    a1.set_ylim(0, .95)
    a1.legend(loc="lower right")
    a1.set_title("El orden pasa de 0,44 a 0,77 con cinco tareas" + NL +
                 "y la dispersion se recupera a medias")
    x = np.arange(len(r))
    a2.bar(x, r.PRED25, .36, color=TINTA, label="dentro del +-25 %")
    a2.bar(x + .38, r.factor2, .36, color=AMBAR, label="dentro del factor 2")
    for i, (p, sig) in enumerate(zip(r.PRED25, r.sorteos_signif)):
        if i and sig >= .8:
            a2.text(i, p + .02, "*", ha="center", fontsize=19, color=AVISO)
    a2.set_xticks(x + .19)
    a2.set_xticklabels(["en frio" if k == 0 else str(int(k)) for k in r.k])
    a2.set_xlabel("tareas del proyecto ya cerradas")
    a2.set_ylim(0, 1.0)
    a2.legend(loc="upper left", ncol=2)
    a2.set_title("Y la mejora del total es significativa desde dos" + NL +
                 "(* = Wilcoxon < 0,05 en al menos el 80 % de los sorteos)")
    fig.suptitle("Lo unico que arregla la escala, y se puede demostrar",
                 x=.005, ha="left", fontsize=15, weight="600")
    guardar(fig, "10_anclas_escala")



def robustez():
    """Las dos comprobaciones que cierran el argumento (E70 y E71).

    Izquierda: el anclaje supone que las tareas ya cerradas son un sorteo al azar.
    Con el ORDEN REAL de cierre no lo es y aun asi no hay sesgo; el escenario que
    si lo rompe (despachar primero lo mas corto) queda acotado y medido.

    Derecha: la compresion de la escala no es un artefacto del corpus. En 186
    proyectos de Apache, independientes, es incluso mas severa.
    """
    o = pd.read_csv("results/orden_cierre/resumen.csv")
    ap = pd.read_json("results/replica_apache/meta.json", typ="series")
    ex = pd.read_csv("results/escala_con_apache/resumen.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

    reglas = [("a · al azar", "lo que supone la curva", TINTA, "o-"),
              ("d · el orden real de cierre", "el orden REAL de cierre", AMBAR, "s-"),
              ("b · las mas cortas de verdad", "lo mas corto primero", AVISO, "v--"),
              ("e · las mas largas primero", "lo mas largo primero", GRIS, "^--")]
    ks = sorted(o[o.k > 0].k.unique())
    a1.axhline(0, color=NEGRO, lw=1.1)
    for clave, etq, col, est in reglas:
        q = o[o.regla.eq(clave)].set_index("k").reindex(ks)
        a1.plot(range(len(ks)), q.SSPB_pc, est, color=col, lw=2.2, ms=7, label=etq)
    a1.set_xticks(range(len(ks)))
    a1.set_xticklabels([str(int(k)) for k in ks])
    a1.set_xlabel("tareas del proyecto ya cerradas")
    a1.set_ylabel("sesgo del total (SSPB, %)")
    a1.legend(loc="lower left", ncol=2, fontsize=10)
    d3 = float(o[(o.k == 3) & o.regla.str.startswith("d")].SSPB_pc.iloc[0])
    a3 = float(o[(o.k == 3) & o.regla.str.startswith("a")].SSPB_pc.iloc[0])
    a1.set_title(f"El orden REAL no mete sesgo: {d3:+.1f} % con tres{NL}"
                 f"cerradas, frente a {a3:+.1f} % si fuesen al azar".replace(".", ","),
                 fontsize=12.5)

    # la compresion, en los dos corpus
    sdA = float(ex[ex.brazo.str.startswith("A")].sd.iloc[0])
    sdR = float(ex[ex.brazo.str.startswith("REAL")].sd.iloc[0])
    spA = float(ex[ex.brazo.str.startswith("A")].spearman.iloc[0])
    corpus = [("los 114 del corpus", sdA, sdR, spA),
              ("los 186 de Apache", float(ap.sd_A), float(ap.sd_real),
               float(ap.spearman_A))]
    x = np.arange(len(corpus))
    a2.bar(x - .19, [c[1] for c in corpus], .36, color=TINTA,
           label="dispersion que el modelo predice")
    a2.bar(x + .19, [c[2] for c in corpus], .36, color=AVISO,
           label="dispersion real")
    for i, (_, sp, sr, sc) in enumerate(corpus):
        a2.text(i - .19, sp + .03, f"{sp:.2f}".replace(".", ","), ha="center",
                fontsize=11, color=TINTA)
        a2.text(i + .19, sr + .03, f"{sr:.2f}".replace(".", ","), ha="center",
                fontsize=11, color=AVISO)
        a2.text(i, -.135, f"cubre el {sp / sr:.0%} · orden {sc:.2f}".replace(".", ","),
                ha="center", fontsize=10.5, color=GRIS)
    a2.set_xticks(x)
    a2.set_xticklabels([c[0] for c in corpus])
    a2.set_ylim(0, 1.25)
    a2.legend(loc="upper left", fontsize=10.5)
    a2.set_title(f"Se replica en un corpus independiente, y es peor:{NL}"
                 f"alli solo cubre el {float(ap.sd_A) / float(ap.sd_real):.0%} "
                 f"del recorrido real", fontsize=12.5)
    fig.suptitle("Las dos comprobaciones que cierran el argumento",
                 x=.005, ha="left", fontsize=15, weight="600")
    guardar(fig, "11_robustez")


if __name__ == "__main__":
    for f in (la_trampa, extremos, frontera, canales, intervalos, panel, analogia,
              total_proyecto, compresion_escala, anclas_escala, robustez):
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} FALLO: {type(e).__name__}: {e}")
