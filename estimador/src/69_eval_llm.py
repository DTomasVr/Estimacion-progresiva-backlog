"""Score every LLM run against our model on the same frozen tasks.

Each (model, condition) is scored four ways, and all four are reported because
each answers a different question:
  crudo      what the LLM says, as is
  calibrado  after one multiplicative factor learned on the SEPARATE calibration
             set -- the fix that removed 44% of the error in the pilot
  combinado  geometric mean of the calibrated LLM and our model for that
             condition, since the two made different errors in the pilot
  validas    crudo restricted to parseable answers, next to the validity rate

Invalid answers are not dropped from the headline numbers: they are replaced by
the text-blind baseline of that condition (global median in A/B, history median
in C). A model that refuses a third of the tasks must not look accurate on the
two thirds it chose to answer.

Differences against our model are tested pairwise on the same tasks: Wilcoxon on
the per-task absolute log error, plus a bootstrap interval for the difference in
MdMRE. The contamination probe reports how often a model names the exact issue
key of a public task it was never told the key of.

This is the September bench (local models), kept as context; the paper's Table III
comes from 261_tablas_llm_api.py, which reuses metricas() and bootstrap_dif() from
here. The CSV files written here (results/llm_bench/resumen.csv,
sesgo_por_tamano.csv, sonda_contaminacion.csv) keep the Spanish column and label
names they were published with: condicion, modelo, validas, seg_tarea, veredicto
("LLM mejor" / "nuestro mejor" / "empate" = LLM better / ours better / tie), etc.
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
BENCH = Path("data/processed/llm_bench")
RAW = Path("results/llm_bench/raw")
CORTES = [0, 1, 4, 16, np.inf]
ETQ = ["<1h", "1-4h", "4-16h", ">16h"]


def sa(y, p, corridas=500, semilla=0):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mar = np.mean(np.abs(y - p))
    rng, n = np.random.default_rng(semilla), len(y)
    base = []
    for _ in range(corridas):
        i = rng.integers(0, n - 1, n)
        i += i >= np.arange(n)
        base.append(np.mean(np.abs(y - y[i])))
    return 1 - mar / np.mean(base)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), 0.01, None)
    m = E.point_metrics(y, p)
    return dict(MdMRE=m["MdMRE"], PRED25=m["PRED25"], PRED50=m["PRED50"],
                MAE_h=m["MAE_h"],
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                SA=sa(y, p))


def bootstrap_dif(y, a, b, n=2000, semilla=1):
    """95% interval for MdMRE(a) - MdMRE(b) on the same tasks."""
    rng = np.random.default_rng(semilla)
    ra, rb = np.abs(y - a) / y, np.abs(y - b) / y
    d = [np.median(ra[i]) - np.median(rb[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(d, [2.5, 97.5])


def clave_real(task_id):
    if task_id.startswith("TAWOS-"):
        return task_id[6:]
    if task_id.startswith("JP-"):
        return task_id.split("-", 2)[2]
    return None


def cargar_corridas():
    filas = []
    for p in glob.glob(str(RAW / "*.jsonl")):
        conjunto = Path(p).name.split("__")[0]
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:   # line half-written during the run
                continue
            r["conjunto"] = conjunto
            filas.append(r)
    return pd.DataFrame(filas)


def main():
    muestra = pd.read_parquet(BENCH / "muestra.parquet")
    cal = pd.read_parquet(BENCH / "calibracion.parquet")
    runs = cargar_corridas()
    if runs.empty:
        sys.exit("no runs in results/llm_bench/raw")

    resumen, sesgos = [], []
    # ------------------------------------------------ our own references
    for cond in ("A", "B", "C"):
        d = muestra if cond != "C" else muestra[muestra.n_hist >= 3]
        y = d.effort_hours.to_numpy()
        refs = ([("constante: mediana global", d.mediana_global),
                 ("nuestro modelo · solo texto", d.modelo_solo_texto),
                 ("nuestro modelo · completo", d.modelo_abs)] if cond != "C" else
                [("mediana del historial", d.mediana_historial),
                 ("nuestro modelo · razón × historial", d.modelo_razon_C)])
        for nombre, p in refs:
            resumen.append(dict(condicion=cond, modelo=nombre, version="referencia",
                                n=len(d), validas=1.0, **metricas(y, p.to_numpy())))

    # ------------------------------------------------ each LLM
    est = runs[runs.condicion != "P"]
    for (backend, modelo, cond), g in est[est.conjunto == "muestra"].groupby(
            ["backend", "modelo", "condicion"]):
        d = muestra if cond != "C" else muestra[muestra.n_hist >= 3]
        d = d.merge(g[["task_id", "valido", "horas", "minimo", "maximo",
                       "segundos", "entrada", "salida"]], on="task_id", how="inner")
        if len(d) < 30:
            continue
        y = d.effort_hours.to_numpy()
        base = (d.mediana_global if cond != "C" else d.mediana_historial).to_numpy()
        nuestro = (d.modelo_abs if cond != "C" else d.modelo_razon_C).to_numpy()
        ok = d.valido.fillna(False).astype(bool).to_numpy()
        crudo = np.where(ok, d.horas.fillna(0).to_numpy(), base)

        # calibration factor learnt ONLY on the calibration set
        gc = runs[(runs.conjunto == "calibracion") & (runs.backend == backend)
                  & (runs.modelo == modelo) & (runs.condicion == cond)]
        cc = cal.merge(gc[gc.valido == True][["task_id", "horas"]], on="task_id")  # noqa: E712
        k = (float(np.exp(np.median(np.log(cc.effort_hours / cc.horas))))
             if len(cc) >= 20 else np.nan)
        calibrado = crudo * (k if k == k else 1.0)
        combinado = np.sqrt(np.clip(calibrado, .01, None) * nuestro)

        etiqueta = f"{modelo} ({backend})"
        # reference on EXACTLY the same tasks: with a half-finished run the
        # figures on the whole sample are not comparable
        resumen.append(dict(condicion=cond, version="referencia", n=len(d),
                            modelo=f"nuestro modelo · mismas {len(d)} tareas que {modelo}",
                            validas=1.0, **metricas(y, nuestro)))
        comun = dict(condicion=cond, modelo=etiqueta, n=len(d), validas=ok.mean(),
                     seg_tarea=float(d.segundos.median()),
                     tokens_entrada=float(d.entrada.median()))
        for version, p in (("crudo", crudo), ("calibrado", calibrado),
                           ("combinado", combinado)):
            err_llm = np.abs(np.log(y) - np.log(np.clip(p, .01, None)))
            err_nos = np.abs(np.log(y) - np.log(nuestro))
            w = wilcoxon(err_llm, err_nos)
            lo, hi = bootstrap_dif(y, p, nuestro)
            # the p-value only says there is a difference; the direction and the verdict
            # say who wins. Verdict on the MdMRE interval, the headline metric
            veredicto = ("LLM mejor" if hi < 0 else "nuestro mejor" if lo > 0 else "empate")
            resumen.append(dict(**comun, version=version,
                                factor=k if version != "crudo" else np.nan,
                                **metricas(y, p),
                                p_wilcoxon=w.pvalue,
                                dif_logerr_mediana=float(np.median(err_llm - err_nos)),
                                dif_MdMRE_IC95=f"[{lo:+.3f}, {hi:+.3f}]",
                                veredicto=veredicto))
        if ok.sum() >= 30:
            resumen.append(dict(**comun, version="solo válidas",
                                **metricas(y[ok], crudo[ok])))
            lo_, hi_ = d.minimo.to_numpy(), d.maximo.to_numpy()
            m_int = ok & np.isfinite(lo_) & np.isfinite(hi_) & (hi_ >= lo_) & (lo_ > 0)
            if m_int.sum() >= 30:
                resumen[-1]["cobertura_80"] = float(np.mean(
                    (y[m_int] >= lo_[m_int]) & (y[m_int] <= hi_[m_int])))
                resumen[-1]["banda_mediana"] = float(np.median(hi_[m_int] / lo_[m_int]))

        tramo = pd.cut(d.effort_hours, CORTES, labels=ETQ)
        for t in ETQ:
            s = (tramo == t).to_numpy() & ok
            if s.sum() >= 5:
                sesgos.append(dict(condicion=cond, modelo=etiqueta, tramo=t,
                                   n=int(s.sum()),
                                   sesgo_llm=float(np.median(crudo[s] / y[s])),
                                   sesgo_nuestro=float(np.median(nuestro[s] / y[s]))))
        for fuente in ("SiP", "TAWOS", "JiraPublico"):
            s = (d.source == fuente).to_numpy()
            if s.sum() >= 30:
                mm = metricas(y[s], calibrado[s])
                resumen.append(dict(condicion=cond, modelo=etiqueta,
                                    version=f"calibrado · {fuente}", n=int(s.sum()),
                                    **mm))

    r = pd.DataFrame(resumen)
    out = Path("results/llm_bench")
    r.to_csv(out / "resumen.csv", index=False)
    pd.DataFrame(sesgos).to_csv(out / "sesgo_por_tamano.csv", index=False)

    cols = ["condicion", "modelo", "version", "n", "validas", "MdMRE", "PRED25",
            "MAE_h", "spearman", "SA", "factor", "p_wilcoxon", "dif_logerr_mediana",
            "dif_MdMRE_IC95", "veredicto", "seg_tarea"]
    vista = r[r.version.isin(["referencia", "crudo", "calibrado", "combinado"])]
    vista = vista[[c for c in cols if c in vista.columns]]
    for cond in ("A", "B", "C"):
        v = vista[vista.condicion == cond].sort_values("MdMRE")
        print(f"\n{'=' * 150}\nCONDITION {cond}\n{'=' * 150}")
        print(v.round(4).to_string(index=False))

    # ------------------------------------------------ contamination probe
    pr = runs[(runs.condicion == "P") & (runs.conjunto == "muestra")]
    if len(pr):
        pr = pr.assign(real=pr.task_id.map(clave_real))
        pr["acierta"] = [isinstance(c, str) and r_ is not None
                         and c.strip().upper() == r_.upper()
                         for c, r_ in zip(pr.clave, pr.real)]
        pr["afirma"] = pr.clave.map(lambda c: isinstance(c, str) and len(c) > 2)
        s = pr.groupby(["backend", "modelo"]).agg(
            n=("task_id", "size"), afirma_reconocer=("afirma", "mean"),
            clave_exacta=("acierta", "mean")).round(4)
        print(f"\n{'=' * 150}\nCONTAMINATION PROBE (public tasks)\n{'=' * 150}")
        print(s.to_string())
        s.to_csv(out / "sonda_contaminacion.csv")

    print(f"\nwritten {out}/resumen.csv, sesgo_por_tamano.csv")


if __name__ == "__main__":
    main()
