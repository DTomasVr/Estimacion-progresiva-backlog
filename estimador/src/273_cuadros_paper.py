"""E-273 - Los cuadros del paper que comparan la tuberia entera con LLM.

Ninguna cifra se escribe a mano: cada cuadro sale de los CSV y JSONL de results/ y
se deja como fragmento LaTeX en paper/cuadros/, que el .tex incluye con \\input.

  tab_llm_estimacion   por tarea, 601 tareas no vistas (results/llm_bench/api_resumen.csv)
  (total por proyecto: solo CSV, hay muy pocos proyectos en el banco para un cuadro)
  tab_llm_generacion   backlog generado, 64 proyectos retenidos (results/banco_generacion)

Para el total, al LLM se le aplica un factor aprendido en las 130 tareas de
calibracion como cociente de SUMAS (el que corresponde a un total), y PBE usa su
cabeza de la media, en frio.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path("data/processed/llm_bench")
RAW = Path("results/llm_bench/raw")
OUT = Path("results/cuadros_paper")
# los fragmentos LaTeX van junto a los CSV; CUADROS_TEX los manda a la carpeta del articulo
TEX = Path(os.environ.get("CUADROS_TEX", OUT / "tex"))
API = {"openai/gpt-oss-120b": "GPT-OSS 120B", "openai/gpt-oss-20b": "GPT-OSS 20B", "qwen/qwen3.8-27b": "Qwen 3.8 27B"}
NOMBRE = {"Qwen 3.8 27B": "Qwen3.8-27B"}   # nombre de la ficha del modelo (guia V2, cambio 9)
MIN_TAREAS = 8

f = lambda x, d=3: "--" if pd.isna(x) else f"{x:.{d}f}".replace(".", "{,}")
pc = lambda x, d=1: "--" if pd.isna(x) else f"{100 * x:.{d}f}".replace(".", "{,}") + r"\,\%"


def tabla(nombre, titulo, cols, cab, filas, nota="", sep="3pt"):
    L = [r"\begin{table}[t]", r"\centering", rf"\caption{{{titulo}}}", rf"\label{{tab:{nombre}}}",
         r"\footnotesize", rf"\setlength{{\tabcolsep}}{{{sep}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
         " & ".join(cab) + r" \\", r"\midrule"]
    for x in filas:
        L.append(x if isinstance(x, str) else " & ".join(x) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if nota:
        L += [r"\par\vspace{2pt}\raggedright\scriptsize " + nota]
    L += [r"\end{table}", ""]
    (TEX / f"tab_{nombre}.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"escrito {TEX / f'tab_{nombre}.tex'}")


def jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])


def estimacion():
    r = pd.read_csv("results/llm_bench/api_resumen.csv")
    r.to_csv(OUT / "estimacion_por_tarea.csv", index=False)
    v7 = r[r.modelo == "PBE v7 (nuestro)"].iloc[0]
    g = lambda x, d=3: "--" if pd.isna(x) else ("$-$" if x < 0 else "") + f(abs(x), d)
    ic = lambda s: s.strip("[]").replace(".", "{,}").replace("-", "$-$").replace(", ", ";\\,")
    filas = [[r"\textbf{Propuesta}", f(v7.MdMRE), pc(v7.PRED25), f(v7.spearman, 2), "--"]]
    a = r[(r.condicion == "A") & r.proveedor.eq("groq")]
    n = {int(v7.n)}
    for nombre in API.values():
        filas.append(r"\midrule")
        for version in ("crudo", "calibrado"):
            x = a[a.modelo.str.startswith(nombre) & (a.version == version)]
            if x.empty:
                continue
            x = x.iloc[0]
            n.add(int(x.n))
            filas.append([NOMBRE.get(nombre, nombre) if version == "crudo" else r"\quad calibrado", f(x.MdMRE), pc(x.PRED25),
                          f(x.spearman, 2), "[" + ic(x.dif_MdMRE) + "]"])
    assert len(n) == 1, n
    tabla("llm_estimacion",
          f"Estimación por tarea frente a LLM en {n.pop()} tareas no usadas para entrenar. Última columna: IC95 de "
          "la diferencia de MdMRE (LLM menos propuesta).",
          "lcccc", ["", "MdMRE", "PRED(25)", r"$\rho$", r"IC95 $\Delta$MdMRE"], filas)


def total():
    mu = pd.read_parquet(BENCH / "muestra.parquet")
    cal = pd.read_parquet(BENCH / "calibracion.parquet")
    cache = BENCH / "v7_media_muestra.parquet"
    if not cache.exists():
        s = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
        e6 = importlib.util.module_from_spec(s)
        s.loader.exec_module(e6)
        est = e6.EstimadorV6(paquete=Path("models/estimador_pbe_v7.joblib"), dispositivo="cpu")
        t = pd.read_parquet("data/processed/pbe_effort_v4.parquet", columns=["task_id", "title"]).drop_duplicates("task_id")
        d = mu[["task_id"]].merge(t, on="task_id", how="left")
        d["v7_media"] = est._media_tarea(d.title.fillna("").astype(str).tolist())
        d[["task_id", "v7_media"]].to_parquet(cache)
    mu = mu.merge(pd.read_parquet(cache), on="task_id")
    tam = mu.groupby("project").size()
    proys = tam[tam >= MIN_TAREAS].index
    # el bloque externo del banco son proyectos de Apache, que la evaluacion por proyecto
    # excluye siempre (horas registradas anomalas: el modelo las sobreestima x1,4 a x14)
    base = mu[mu.project.isin(proys) & (mu.bloque == "prueba_d1")]
    proys = pd.Index(base.project.unique())
    filas, salida = [], []

    def mide(nombre, version, d, col):
        g = d.groupby("project")
        t, r = g[col].sum(), g.effort_hours.sum()
        e = np.abs(t - r) / r
        m = dict(sistema=nombre, version=version, proyectos=len(r), tareas=len(d), PRED25=float(np.mean(e <= .25)),
                 MdMRE=float(np.median(e)), factor2=float(np.mean(np.maximum(t / r, r / t) <= 2)))
        salida.append(m)
        return m

    m = mide("PBE", "en frío", base, "v7_media")
    filas.append([r"\textbf{Propuesta}", "en frío", str(m["proyectos"]), pc(m["PRED25"]), f(m["MdMRE"]), pc(m["factor2"])])
    for modelo, nombre in API.items():
        slug = re.sub(r"[^\w.-]", "_", modelo)
        pm, pcal = RAW / f"muestra__groq__{slug}__A.jsonl", RAW / f"calibracion__groq__{slug}__A.jsonl"
        if not pm.exists():
            continue
        g = jsonl(pm)
        d = base.merge(g[["task_id", "valido", "horas"]], on="task_id")
        d["llm"] = np.where(d.valido.fillna(False).astype(bool), d.horas.fillna(0), d.mediana_global)
        completos = d.groupby("project").size().reindex(proys).fillna(0) == tam[proys]
        d = d[d.project.isin(completos[completos].index)]
        c = cal.merge(jsonl(pcal).query("valido == True")[["task_id", "horas"]], on="task_id")
        k = c.effort_hours.sum() / c.horas.sum()
        d["llm_cal"] = d.llm * k
        filas.append(r"\midrule")
        for version, col in (("sin calibrar", "llm"), ("calibrado", "llm_cal")):
            m = mide(nombre, version, d, col)
            filas.append([nombre if version == "sin calibrar" else "", version, str(m["proyectos"]), pc(m["PRED25"]),
                          f(m["MdMRE"]), pc(m["factor2"])])
        m = mide("PBE (mismos proyectos que " + nombre + ")", "en frío", d, "v7_media")
    pd.DataFrame(salida).to_csv(OUT / "total_por_proyecto.csv", index=False)
    # Con tan pocos proyectos no se saca cuadro para el paper: queda solo el CSV.
    print(pd.DataFrame(salida).round(3).to_string(index=False))


def generacion():
    p = Path("results/banco_generacion/resumen.csv")
    if not p.exists():
        print("falta results/banco_generacion/resumen.csv: corre antes src/272_banco_generacion.py tablas")
        return
    r = pd.read_csv(p)
    red = pd.read_csv("results/redundancia_tareas/resumen.csv").set_index("sistema").casi_duplicadas_090
    red.index = [("PBE (actual)" if s == "pbe" else s + " cero") for s in red.index]
    r["casi_duplicadas"] = r.sistema.map(red)
    r.to_csv(OUT / "generacion.csv", index=False)
    orden = [("PBE (actual)", r"\textbf{Propuesta}"), ("gemma2:9b cero", "Gemma 2 9B"),
             ("llama3.1:8b cero", "Llama 3.1 8B"), ("mistral:7b cero", "Mistral 7B")]
    filas, ns = [], set()
    for clave, nombre in orden:
        x = r[r.sistema == clave]
        if x.empty or x.iloc[0].n < 60:      # una corrida a medias no entra en el cuadro
            continue
        x = x.iloc[0]
        ns.add(int(x.n))
        filas.append([nombre, f(x.sF1), pc(x.cobertura), f(x.tareas, 0), pc(x.casi_duplicadas),
                      f(x.seg_mediana, 1), "--" if clave == "PBE (actual)" else pc(x.reconoce, 0)])
    assert len(ns) == 1, ns
    tabla("llm_generacion",
          f"Generación frente a LLM en {ns.pop()} proyectos retenidos. Duplicadas: coseno $\\geq 0{{,}}90$ con otra "
          "tarea; s: segundos por backlog en la misma GPU; Reconoce: proyectos que el LLM dice reconocer, solo un "
          "indicio de contaminación.",
          "lcccccc", ["", "sF1", "Cobertura", "Tareas", "Duplicadas", "s", "Reconoce"], filas, sep="2.5pt")


CASO = "dalpiaz:g02-federalspending"


def lado():
    """Un mismo proyecto, la primera historia y sus tres primeras tareas en cada sistema.

    El caso lo eligio el chat de generacion (CUADROS_GENERACION_PAPER.md): el de Dalpiaz
    mas legible, donde PBE y GPT-OSS empatan en sF1. El texto sale de los JSONL."""
    esc = lambda s: re.sub(r"([&%$#_])", r"\\\1", s)
    gen = Path("F:/Software Task generation Model/results/20_banco.jsonl")
    R = [json.loads(l) for l in open(gen, encoding="utf-8")]
    R = [r for r in R if r["sistema"] != "pbe"]
    for p in sorted(Path("results/banco_generacion").glob("*.jsonl")):
        R += [json.loads(l) for l in open(p, encoding="utf-8")]
    R = {(r["sistema"], r["condicion"]): r for r in R if r["caso"] == CASO}
    pp = pd.read_csv("results/banco_generacion/por_proyecto.csv")
    pp = pp[pp.caso == CASO].set_index("sistema")
    s = importlib.util.spec_from_file_location("b272", "src/272_banco_generacion.py")
    b272 = importlib.util.module_from_spec(s)
    s.loader.exec_module(b272)
    C = b272.casos()
    entrada = " ".join(C[C.parent_uid == CASO].entrada.iloc[0].split())
    orden = [(("pbe", "actual"), "PBE (actual)", r"\textbf{Propuesta}"),
             (("openai/gpt-oss-120b", "cero"), "openai/gpt-oss-120b cero", "GPT-OSS 120B"),
             (("gemma2:9b", "cero"), "gemma2:9b cero", "Gemma 2 9B"),
             (("llama3.1:8b", "cero"), "llama3.1:8b cero", "Llama 3.1 8B"),
             (("mistral:7b", "cero"), "mistral:7b cero", "Mistral 7B")]
    filas, salida = [], []
    for clave, sis, nombre in orden:
        if clave not in R or sis not in pp.index:
            continue
        r, m = R[clave], pp.loc[sis]
        tareas = "; ".join(r["tareas"][0][:3])
        filas.append([nombre, esc(r["historias"][0]), esc(tareas), str(int(m.n_tareas)), f(m.sF1), f(m.segundos, 1)])
        salida.append(dict(sistema=sis, historia=r["historias"][0], tareas=tareas, n_tareas=int(m.n_tareas),
                           sF1=float(m.sF1), segundos=float(m.segundos)))
    pd.DataFrame(salida).assign(entrada=entrada).to_csv(OUT / "lado_a_lado.csv", index=False)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{El mismo proyecto retenido (portal de gasto federal, del conjunto de Dalpiaz) en cada sistema: "
         r"la primera historia generada y sus tres primeras tareas. «Tareas» es el total del backlog; el sF1 solo "
         r"puntúa las historias. Todos los sistemas reciben exactamente la misma entrada, que es esta descripción: "
         r"\emph{" + esc(entrada) + r"} Las salidas se muestran en inglés, como las produce cada sistema.}",
         r"\label{tab:lado}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l p{0.33\textwidth} p{0.36\textwidth} c c c}", r"\toprule",
         r" & Primera historia & Primeras tareas & Tareas & sF1 & s \\", r"\midrule"]
    L += [" & ".join(x) + r" \\" for x in filas]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TEX / "tab_lado.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"escrito {TEX / 'tab_lado.tex'}")


BANDA = {1: "D · tramo contraido M=1500", 5: "D · tramo contraido M=200",
         10: "A2 · por k y tramo (hoy)", 20: "A2 · por k y tramo (hoy)"}


def anclaje():
    """El total segun las tareas cerradas: la curva que es el centro de la propuesta.

    Sale de E-275 (results/curva_anclaje_dos_criterios). La anchura del intervalo con
    k >= 1 es la de la variante desplegada para cada k; en el arranque (k = 0), la del
    intervalo unico del 80 % de results/banda_proyecto."""
    c = pd.read_csv("results/curva_anclaje_dos_criterios/resumen.csv")
    b = pd.read_csv("results/banda_que_cierra_antes/resumen.csv")
    b0 = pd.read_csv("results/banda_proyecto/resumen.csv")
    b0 = b0[(b0.escenario == "en frio") & (b0.nivel == "80%") & (b0.metodo == "unico")].iloc[0]
    v = lambda k, crit, col: float(c[(c.k == k) & (c.criterio == crit)][col].iloc[0])
    filas, salida = [], []
    for k in (0, 10, 20):      # guia V3: la fila k = 1 va en el texto (recorte de reserva)
        if k == 0:
            ancho, cob = float(b0.anchura_x), float(b0.cobertura)
        else:
            x = b[(b.k == k) & (b.banda == BANDA[k])].iloc[0]
            ancho, cob = float(x.anchura_mediana), float(x.cobertura)
        s = dict(cerradas=k, PRED25=v(k, "todos", "PRED25"), factor2=v(k, "todos", "factor2"),
                 MdMRE=v(k, "todos", "MdMRE"), intervalo=ancho, cobertura=cob)
        if k:
            s.update(PRED25_mayores=v(k, "mayores", "PRED25"), proyectos_mayores=int(v(k, "mayores", "proyectos")),
                     frio_mayores=v(k, "mayores", "PRED25_en_frio"),
                     PRED25_regla=v(k, "k o el 20 % del backlog", "PRED25"))
        salida.append(s)
        frio_todos = pc(v(1, "mayores", "PRED25_en_frio"))
        mayores = (frio_todos + f" (114; {frio_todos})" if not k
                   else pc(s["PRED25_mayores"]) + f" ({s['proyectos_mayores']}; {pc(s['frio_mayores'])})")
        filas.append(["0" if k == 0 else str(k), mayores, pc(s["PRED25"]), pc(s["factor2"]), f(s["MdMRE"]),
                      r"$\times$" + f(ancho, 2), f(cob, 2)])
    pd.DataFrame(salida).to_csv(OUT / "anclaje.csv", index=False)
    tabla("anclaje",
          "El total del proyecto según las tareas cerradas, con cada proyecto predicho como nuevo, sobre los 114 "
          "proyectos (incluye los de $k$ o menos tareas, ya cerrados). «Solo mayores»: proyectos con más de $k$ "
          "tareas (cuántos; PRED(25) al arrancar). MdMRE: mediana del error relativo; $\\times$2: dentro de un "
          "factor 2; Cob.: cobertura del intervalo.",
          "lcccccc", ["Cerradas", "Solo mayores", "PRED(25)", r"$\times$2", "MdMRE", "Interv.", "Cob."], filas,
          sep="2pt")


def tres_estados():
    """Cuanto estima el usuario x cuantas tareas hay cerradas, con UNA sola configuracion."""
    r = pd.read_csv("results/tres_estados/resumen.csv")
    meta = json.load(open("results/tres_estados/meta.json"))
    ks = (0, 3, 5, 10, 20)

    def celda(est, k):
        x = r[(r.usuario_estima == est) & (r.cerradas == k)]
        if est != "0 %":
            x = x[x.lambda_u == 12]
        if k:
            x = x[x.lambda_c == 30]
        assert len(x) == 1, (est, k, len(x))
        return float(x.PRED25.iloc[0])

    filas, salida = [], []
    for est, nombre in (("0 %", "nada"), ("100 %", "todo el backlog")):
        v = [celda(est, k) for k in ks]
        filas.append([nombre] + [pc(x) for x in v])
        salida.append(dict(usuario_estima=est, **{f"k{k}": x for k, x in zip(ks, v)}))
    pd.DataFrame(salida).to_csv(OUT / "tres_estados.csv", index=False)
    tabla("tres_estados",
          f"PRED(25) del total cuando el usuario aporta sus estimaciones al arrancar y luego se cierran tareas "
          f"({meta['proyectos']} proyectos con estimaciones registradas; $\\lambda_u = 12$, $\\lambda'_c = 30$).",
          "lccccc", ["El usuario estima"] + [f"{k} cerr." for k in ks], filas)


def respuestas():
    """Lo que el usuario contesta al arrancar, tratado como tareas cerradas."""
    r = pd.read_csv("results/la_ultima_vez/como_cerradas.csv")
    frio = pd.read_csv("results/la_ultima_vez/previo_del_equipo.csv")
    frio = frio[(frio.cerradas == 0) & (frio.w == 0)].iloc[0]
    meta = json.load(open("results/la_ultima_vez/meta.json"))
    x = r[r.respuesta == "propia (acierta)"].set_index("tareas_preguntadas")
    filas = [["0 (arranque)", pc(frio.PRED25), pc(frio.factor2), f(frio.MdMRE)]]
    filas += [[str(m), pc(x.loc[m].PRED25), pc(x.loc[m].factor2), f(x.loc[m].MdMRE)] for m in (1, 3, 5, 10, 20)]
    x.reset_index().to_csv(OUT / "respuestas.csv", index=False)
    tabla("respuestas",
          f"El total cuando, al arrancar, el usuario da las horas de algunas tareas del backlog y estas se tratan "
          f"como cerradas ({meta['proyectos']} proyectos). Cota superior: el usuario acierta las horas.",
          "lccc", ["Tareas que da el usuario", "PRED(25)", r"$\times$2", "MdMRE"], filas)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    if "--articulo" in sys.argv:          # solo las tablas del articulo, desde los CSV de results/
        estimacion()
        generacion()
        anclaje()
        sys.exit()
    estimacion()
    total()
    generacion()
    lado()
    anclaje()
    tres_estados()
    respuestas()
