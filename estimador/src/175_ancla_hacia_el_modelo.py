"""E68 - Contraer el anclaje HACIA EL MODELO DE ESCALA, no hacia cero.

Las dos vias que se han probado para la compresion tienen fortalezas opuestas:

  E66, modelo de escala del proyecto   funciona sin ninguna tarea cerrada, sube el
                                       orden de 0,444 a 0,481, pero la mejora en el
                                       total NO es significativa (Wilcoxon p = 0,44)
  E67, anclar con tareas cerradas      sube el orden a 0,700 con tres tareas y a
                                       0,769 con cinco, y es significativo en el
                                       100 % de los sorteos desde k = 3. Pero con
                                       cero tareas no existe, y con una o dos es
                                       ruidoso

Y hay una forma obvia de juntarlas que no se ha probado. Ahora mismo el
desplazamiento observado se contrae hacia CERO:

    off = off_observado x k / (k + lambda)

Contraer hacia cero significa "si tengo pocas tareas, me quedo con lo que dice el
modelo en frio". Pero el modelo en frio es justo el que esta comprimido. Si en
lugar de cero se contrae hacia el desplazamiento que sugiere el MODELO DE ESCALA:

    off = w x off_observado + (1 - w) x off_modelo,   w = k / (k + lambda)

entonces con cero tareas se usa la escala del modelo, con muchas manda lo
observado, y en medio se mezclan. Es la misma contraccion de James-Stein pero
hacia un centro informado en vez de hacia el origen, que es como se usa cuando
hay un estimador previo decente.

  A  hacia cero            lo que hay
  B  hacia el modelo       la propuesta
  C  hacia el modelo, con lambda distinto para cada centro
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
OUT = Path("results/ancla_modelo")
S = "=" * 122
MIN_TAREAS, SORTEOS, LAMBDA = 8, 60, 12
KS = (0, 1, 2, 3, 5, 10, 20)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    g = d.groupby("p")
    real_tot = g.real.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy().astype(float)
    esc_r = np.log(g.real.mean().reindex(nom).to_numpy())
    frio = g.est.sum().reindex(nom).to_numpy()
    esc_frio = np.log(frio / n_t)
    # el modelo de escala de E66, columna E (vector + agregados + la media), ya
    # calculado dejando fuera un proyecto cada vez
    esc_mod = np.load("results/predecir_escala/predicciones.npy")[:, 4]
    off_mod = esc_mod - esc_frio
    P = len(nom)
    print(f"{P} proyectos · desplazamiento que sugiere el modelo de escala: "
          f"mediana x{np.exp(np.median(off_mod)):.3f}, "
          f"sd {off_mod.std():.3f} · {time.time() - t0:.0f} s", flush=True)

    def corrida(k, centro, lam, rng):
        tot = np.zeros(P)
        for j, p in enumerate(nom):
            i = idx[p]
            if k == 0 or len(i) <= k:
                tot[j] = (es_[i] * np.exp(centro[j])).sum()
                continue
            sel = rng.choice(i, k, replace=False)
            resto = np.setdiff1d(i, sel)
            obs = np.log(re_[sel].sum() / (es_[sel] * np.exp(centro[j])).sum())
            w = k / (k + lam)
            off = centro[j] + w * obs
            tot[j] = re_[sel].sum() + (es_[resto] * np.exp(off)).sum()
        return tot

    def mide(t):
        r = np.abs(t - real_tot) / real_tot
        q = np.log(t / real_tot)
        m = float(np.median(q))
        e = np.log(t / n_t)
        return dict(MdMRE=float(np.median(r)), PRED25=float(np.mean(r <= .25)),
                    factor2=float(np.mean(np.maximum(t / real_tot, real_tot / t) <= 2)),
                    MAE_h=float(np.abs(t - real_tot).mean()),
                    R2_log=float(1 - ((np.log(real_tot) - np.log(t)) ** 2).mean()
                                 / np.log(real_tot).var()),
                    SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                    esc_spearman=float(pd.Series(e).corr(pd.Series(esc_r), method="spearman")),
                    esc_sd=float(e.std()))

    cero = np.zeros(P)
    err0 = np.abs(np.log(frio / real_tot))
    filas = []
    for k in KS:
        for etq, centro, lam in (("A · hacia cero", cero, LAMBDA),
                                 ("B · hacia el modelo de escala", off_mod, LAMBDA),
                                 ("D · hacia el modelo, sin sesgo", off_mod, LAMBDA)):
            rng = np.random.default_rng(8000 + k)
            ms, pw = [], []
            n_s = 1 if k == 0 else SORTEOS
            for _ in range(n_s):
                t = corrida(k, centro, lam, rng)
                if etq.startswith("D"):
                    # El desplazamiento del modelo de escala viene de una regresion
                    # en log, asi que exp() de el es una MEDIANA y deja el total
                    # corto (sesgo -9 %). Se centra con un factor estimado dejando
                    # fuera cada proyecto. Es la sexta vez que aparece esto.
                    q = np.log(t / real_tot)
                    c = np.array([-np.median(np.delete(q, i)) for i in range(P)])
                    t = t * np.exp(c)
                ms.append(mide(t))
                pw.append(float(wilcoxon(np.abs(np.log(t / real_tot)), err0).pvalue)
                          if k else np.nan)
            filas.append(dict(k=k, brazo=etq,
                              **{c: float(np.mean([m[c] for m in ms])) for c in ms[0]},
                              p_vs_frio=float(np.median(pw)) if k else np.nan))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    print(f"\n{S}\nCONTRAER HACIA EL MODELO DE ESCALA EN VEZ DE HACIA CERO\n{S}")
    print(t[["k", "brazo", "MdMRE", "PRED25", "factor2", "MAE_h", "R2_log",
             "SSPB_pc", "esc_spearman", "esc_sd", "p_vs_frio"]].round(4)
          .to_string(index=False))
    print(f"  esc_sd real = {esc_r.std():.3f} · p_vs_frio es la mediana entre sorteos")
    for k in KS:
        v = t[t.k == k]
        if len(v) > 1:
            a = v[v.brazo.str.startswith("A")].iloc[0]
            b = v[v.brazo.str.startswith("B")].iloc[0]
            print(f"  k={k:2d}: MdMRE {a.MdMRE:.4f} -> {b.MdMRE:.4f} · "
                  f"+-25 % {a.PRED25:.4f} -> {b.PRED25:.4f}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"sorteos": SORTEOS, "lambda": LAMBDA},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
