"""E102 - ¿Sobreajusta el EXPERTO EN TAREAS CORTAS? De eso depende lo sintetico.

La pregunta no es retorica y decide la estrategia entera:

  si SOBREAJUSTA     el error de entrenamiento es mucho menor que el de fuera
                     de pliegue -> mas datos (reales o sinteticos) y mas
                     regularizacion ayudan
  si INFRAAJUSTA     los dos errores son parecidos y los dos malos -> el
                     modelo no tiene capacidad o los rasgos no bastan; mas
                     datos NO arreglan nada
  si es RUIDO        los dos parecidos, la curva de aprendizaje plana y mas
                     capacidad tampoco mueve -> el techo es del problema

E90 dejo el experto corto -un regresor entrenado SOLO con tareas de <=2 h- con
un diagnostico raro: arregla la escala de su regimen (sesgo x2,46 -> x0,85,
±25 % 15,5 % -> 20,0 %) pero su spearman DENTRO del regimen es 0,058, o sea que
no ordena. Hay dos explicaciones posibles y llevan a sitios opuestos:

  SOBREAJUSTA   entrena con ~10.000 tareas cortas, memoriza el entrenamiento y
                fuera de pliegue se queda sin nada que decir. Entonces mas
                tareas cortas -sinteticas incluidas- SI arreglarian el orden
  NO HAY SENAL  dentro de "menos de 2 horas" el texto de verdad no distingue
                una tarea de 20 minutos de una de 2 horas. Entonces generar
                tareas cortas sinteticas no anade informacion y no sirve

Se mide con tres cosas que juntas no dejan sitio a la duda, sobre el experto y
sobre el global para comparar:

  1  BRECHA          error dentro del entrenamiento contra fuera de pliegue. Si
                     el experto memoriza, la brecha sera grande
  2  CURVA           entrenar con el 10, 25, 50 y 100 % de los PROYECTOS. Si la
                     curva ya esta plana, anadir datos -sinteticos o no- no
                     cambia nada, y eso se ve antes de generarlos
  3  CAPACIDAD       barrido de profundidad, iteraciones y regularizacion. Si
                     mas regularizacion mejora, sobreajusta; si mas capacidad
                     mejora, infraajusta; si nada mueve, es ruido

Se mira el spearman ademas del error, porque es lo que le falta al experto y lo
que decidiria si vale la pena generar datos.

Objetivo log(horas) con error cuadratico, GroupKFold por proyecto entero.
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/sobreajuste")
S = "=" * 100
PLIEGUES, CORTE = 3, 2.0
FRACS = (.10, .25, .50, 1.0)
BASE = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
            random_state=0)
CAPACIDAD = [
    ("mas regularizado  l2=100", dict(l2_regularization=100.)),
    ("mas regularizado  l2=10", dict(l2_regularization=10.)),
    ("el actual", {}),
    ("mas capacidad  prof=10", dict(max_depth=10)),
    ("mas capacidad  iter=800", dict(max_iter=800)),
    ("mucha capacidad  prof=12 iter=800", dict(max_depth=12, max_iter=800)),
]


def err(y, p):
    """Error absoluto mediano en log: la escala natural del problema."""
    return float(np.median(np.abs(np.log(np.clip(p, 1e-6, None)) - np.log(y))))


def rho(y, p):
    """Spearman: lo que al experto corto le falta."""
    from scipy.stats import spearmanr
    if len(y) < 5 or np.std(p) < 1e-12:
        return float("nan")
    return float(spearmanr(p, y).statistic)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap].astype(np.float32)
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    corto = y <= CORTE
    print(f"{len(y):,} tareas · {sw.project.nunique()} proyectos · "
          f"cortas {corto.mean():.1%} · {time.time() - t0:.0f} s", flush=True)

    part = list(GroupKFold(PLIEGUES).split(V, y, proy))

    def entrena(a_, par, solo_cortas):
        """Ajusta el global o el experto corto sobre el pliegue a_."""
        tr = a_[corto[a_]] if solo_cortas else a_
        return HistGradientBoostingRegressor(**par).fit(V[tr], ly[tr]), tr

    # ---------- 1. la brecha ----------
    print(f"\n{S}\n1. LA BRECHA ENTRE ENTRENAMIENTO Y FUERA DE PLIEGUE\n{S}")
    print(f"  {'modelo':>22}{'evaluado en':>14}{'dentro':>9}{'fuera':>9}"
          f"{'brecha':>9}{'rho fuera':>11}")
    fil_b = []
    a, b = part[0]
    for nombre, solo in (("global", False), ("EXPERTO corto", True)):
        mo, tr = entrena(a, BASE, solo)
        pa, pb = np.exp(mo.predict(V[tr])), np.exp(mo.predict(V[b]))
        amb = (("cortas", corto[tr], corto[b]) if solo else
               ("todas", np.ones(len(tr), bool), np.ones(len(b), bool)))
        pares = [amb] if solo else [amb, ("cortas", corto[tr], corto[b])]
        for etq, ma, mb in pares:
            ea, eb = err(y[tr][ma], pa[ma]), err(y[b][mb], pb[mb])
            r_ = rho(y[b][mb], pb[mb])
            fil_b.append(dict(modelo=nombre, ambito=etq, dentro=ea, fuera=eb,
                              brecha=eb - ea, rho_fuera=r_))
            print(f"  {nombre:>22}{etq:>14}{ea:>9.4f}{eb:>9.4f}{eb - ea:>9.4f}"
                  f"{r_:>11.4f}")
    print(f"  · {time.time() - t0:.0f} s", flush=True)

    # ---------- 2. la curva de aprendizaje, por PROYECTOS ----------
    print(f"\n{S}\n2. CURVA DE APRENDIZAJE DEL EXPERTO CORTO (se quitan "
          f"PROYECTOS enteros)\n{S}")
    print(f"  {'% proyectos':>12}{'n cortas':>11}{'error fuera':>13}"
          f"{'rho fuera':>11}")
    fil_c = []
    for fr in FRACS:
        ee, rr, nn = [], [], []
        for nf, (a_, b_) in enumerate(part):
            ps = pd.unique(proy[a_])
            rng = np.random.default_rng(900 + nf)
            keep = set(rng.choice(ps, max(2, int(len(ps) * fr)), replace=False))
            sub = a_[np.isin(proy[a_], list(keep))]
            sub = sub[corto[sub]]
            nn.append(len(sub))
            mo = HistGradientBoostingRegressor(**BASE).fit(V[sub], ly[sub])
            p_ = np.exp(mo.predict(V[b_][corto[b_]]))
            ee.append(err(y[b_][corto[b_]], p_))
            rr.append(rho(y[b_][corto[b_]], p_))
        fil_c.append(dict(frac=fr, n=int(np.mean(nn)), error=float(np.mean(ee)),
                          rho=float(np.nanmean(rr))))
        print(f"  {fr:>11.0%}{int(np.mean(nn)):>11,}{np.mean(ee):>13.4f}"
              f"{np.nanmean(rr):>11.4f}")
    print(f"  · {time.time() - t0:.0f} s", flush=True)

    # ---------- 3. capacidad ----------
    print(f"\n{S}\n3. ¿MAS REGULARIZACION O MAS CAPACIDAD? (el EXPERTO corto)\n{S}")
    print(f"  {'configuracion':>36}{'dentro':>9}{'fuera':>9}{'brecha':>9}"
          f"{'rho fuera':>11}")
    fil_k = []
    for etq, extra in CAPACIDAD:
        par = dict(BASE, **extra)
        ea, eb, rr = [], [], []
        for a_, b_ in part:
            mo, tr = entrena(a_, par, True)
            pa_ = np.exp(mo.predict(V[tr]))
            pb_ = np.exp(mo.predict(V[b_][corto[b_]]))
            ea.append(err(y[tr], pa_))
            eb.append(err(y[b_][corto[b_]], pb_))
            rr.append(rho(y[b_][corto[b_]], pb_))
        d = dict(config=etq, dentro=float(np.mean(ea)), fuera=float(np.mean(eb)),
                 rho=float(np.nanmean(rr)))
        d["brecha"] = d["fuera"] - d["dentro"]
        fil_k.append(d)
        print(f"  {etq:>36}{d['dentro']:>9.4f}{d['fuera']:>9.4f}"
              f"{d['brecha']:>9.4f}{d['rho']:>11.4f}")
        print(f"      · {time.time() - t0:.0f} s", flush=True)

    # ---------- veredicto ----------
    print(f"\n{S}\nVEREDICTO\n{S}")
    tc = pd.DataFrame(fil_c)
    tk = pd.DataFrame(fil_k)
    act = tk[tk.config == "el actual"].iloc[0]
    mejor = tk.loc[tk.fuera.idxmin()]
    mejor_rho = tk.loc[tk.rho.idxmax()]
    pendiente = float(tc.iloc[-1].error - tc.iloc[-2].error)
    d_rho = float(tc.iloc[-1].rho - tc.iloc[-2].rho)
    brecha = float(act.brecha)
    print(f"  brecha del EXPERTO corto: {brecha:.4f} en log "
          f"(x{np.exp(brecha):.2f} de error extra fuera de pliegue)")
    print(f"  la curva, del 50 % al 100 % de los proyectos: error "
          f"{pendiente:+.4f} · rho {d_rho:+.4f}")
    print(f"  mejor por error: {mejor.config} ({mejor.fuera:.4f} contra "
          f"{act.fuera:.4f})")
    print(f"  mejor por rho:   {mejor_rho.config} ({mejor_rho.rho:.4f} contra "
          f"{act.rho:.4f})")
    print()
    if brecha > .25:
        print("  SOBREAJUSTA. Mas tareas cortas tienen recorrido, y las")
        print("  sinteticas pueden valer si se parecen a las reales.")
    elif abs(pendiente) < .005 and abs(d_rho) < .02 and \
            mejor.fuera >= act.fuera - .005:
        print("  NO SOBREAJUSTA: la brecha es pequena, la curva esta PLANA y mas")
        print("  capacidad no mejora. Dentro del regimen corto el texto no dice")
        print("  cual tarea cuesta mas, y generar tareas cortas sinteticas no")
        print("  anade informacion que no este ya ahi: las etiquetas tendrian que")
        print("  salir de este mismo modelo, que es justo el que no sabe.")
    elif abs(pendiente) >= .005 or abs(d_rho) >= .02:
        print("  La curva TODAVIA se mueve: mas tareas cortas ayudarian, asi que")
        print("  los datos sinteticos merecen una prueba.")
    else:
        print("  Infraajusta: mas capacidad mejora. Hay que mirar el modelo antes")
        print("  que los datos.")
    pd.DataFrame(fil_b).to_csv(OUT / "brecha.csv", index=False)
    tc.to_csv(OUT / "curva.csv", index=False)
    tk.to_csv(OUT / "capacidad.csv", index=False)
    json.dump({"brecha": brecha, "pendiente_50_100": pendiente,
               "mejor_config": str(mejor.config), "fuera_actual": float(act.fuera),
               "fuera_mejor": float(mejor.fuera)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
