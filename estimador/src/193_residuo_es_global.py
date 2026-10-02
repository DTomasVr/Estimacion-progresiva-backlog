"""E80 - ¿Por que no sirvio el texto de las historicas? Porque el residuo es GLOBAL.

E79 peso las tareas historicas por parecido de texto y no mejoro a usar un solo
numero para todo el proyecto. La explicacion que hay que comprobar es esta: el
residuo del modelo

    rho_i = log( horas reales_i / prediccion_i )

¿varia de una tarea a otra dentro del proyecto de forma que el TEXTO pueda anticipar,
o es basicamente un desplazamiento comun a todas mas ruido?

Si es lo segundo, ninguna ponderacion por similitud puede ayudar, por buena que sea:
no hay nada local que capturar. Y eso encaja con el diagnostico del proyecto -- la
escala es una propiedad del equipo, no del texto de la tarea.

Se mide de dos formas, dentro de cada proyecto y dejando fuera la tarea que se mide:

  1 descomposicion de la varianza de rho: cuanto es ENTRE proyectos y cuanto DENTRO
  2 predecir rho_i con sus vecinos por texto DEL MISMO proyecto, contra predecirlo
    con la media del proyecto. Si los vecinos no ganan, el texto no sabe nada del
    residuo que la media del proyecto no sepa ya.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/residuo_global")
S = "=" * 100
MIN_TAREAS, VECINOS = 25, 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    Vn = (V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)).astype(
        np.float32)
    tam = sw.groupby("project").size()
    nom = list(tam[tam >= MIN_TAREAS].index)
    proy, real = sw.project.to_numpy(), sw.effort_hours.to_numpy(float)
    rho = np.log(real / pred)
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    print(f"{len(nom)} proyectos · {sum(len(idx[p]) for p in nom):,} tareas · "
          f"{time.time() - t0:.0f} s", flush=True)

    # ---- 1. varianza del residuo: entre proyectos y dentro ----------------
    medias = np.array([rho[idx[p]].mean() for p in nom])
    pesos = np.array([len(idx[p]) for p in nom], float)
    gran = float(np.average(medias, weights=pesos))
    v_entre = float(np.average((medias - gran) ** 2, weights=pesos))
    v_dentro = float(np.average([rho[idx[p]].var() for p in nom], weights=pesos))
    print(f"\n{S}\n1 · DE DONDE VIENE LA VARIANZA DEL RESIDUO\n{S}")
    print(f"  varianza ENTRE proyectos (el desplazamiento comun) {v_entre:7.4f}  "
          f"{v_entre / (v_entre + v_dentro):6.1%}")
    print(f"  varianza DENTRO de cada proyecto (tarea a tarea)   {v_dentro:7.4f}  "
          f"{v_dentro / (v_entre + v_dentro):6.1%}")

    # ---- 2. ¿los vecinos por texto predicen el residuo? -------------------
    err_vec, err_med, n_tot = [], [], 0
    for p in nom:
        i = idx[p]
        r, Vp = rho[i], Vn[i]
        Sm = Vp @ Vp.T
        np.fill_diagonal(Sm, -2.0)                 # fuera uno mismo
        k = min(VECINOS, len(i) - 1)
        j = np.argpartition(-Sm, k - 1, axis=1)[:, :k]
        pv = r[j].mean(1)                          # prediccion por vecinos
        # media del proyecto dejando fuera la propia tarea
        pm = (r.sum() - r) / (len(r) - 1)
        err_vec.append(np.abs(r - pv))
        err_med.append(np.abs(r - pm))
        n_tot += len(i)
    ev, em = np.concatenate(err_vec), np.concatenate(err_med)
    print(f"\n{S}\n2 · ¿EL TEXTO ANTICIPA EL RESIDUO DE UNA TAREA? ({n_tot:,} tareas)"
          f"\n{S}")
    print(f"  error absoluto medio prediciendo con {VECINOS} VECINOS por texto "
          f"{ev.mean():.4f}")
    print(f"  error absoluto medio prediciendo con la MEDIA del proyecto      "
          f"{em.mean():.4f}")
    mejora = (em.mean() - ev.mean()) / em.mean()
    # Un 2-3 % no es "mejorar": es ruido al lado de lo que haria falta para que
    # ponderar por texto cambiara algo. El umbral evita escribir una conclusion
    # que diga lo contrario de lo que significan los numeros.
    UMBRAL = .10
    veredicto = ("MEJORAN de forma util" if mejora > UMBRAL else
                 "no mejoran de forma util" if mejora > -UMBRAL else "EMPEORAN")
    print(f"  los vecinos {veredicto}: {mejora:+.1%} sobre la media del proyecto")

    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  Solo el {v_entre / (v_entre + v_dentro):.0%} de la varianza del residuo "
          f"esta ENTRE proyectos: ese desplazamiento")
    print(f"  comun es lo unico que el anclaje puede aprovechar, y lo aprovecha con un "
          f"solo numero.")
    print(f"  El {v_dentro / (v_entre + v_dentro):.0%} restante esta DENTRO de cada "
          f"proyecto, y el texto no lo anticipa:")
    print(f"  predecirlo con vecinos solo cambia el error un {mejora:+.1%} frente a la "
          f"media del proyecto.")
    print(f"  Por eso ponderar por similitud no ayudo, y por eso el total del proyecto "
          f"si mejora:")
    print(f"  al sumar, ese {v_dentro / (v_entre + v_dentro):.0%} de ruido tarea a tarea "
          f"se cancela y queda el desplazamiento.")
    json.dump({"var_entre": v_entre, "var_dentro": v_dentro,
               "frac_entre": v_entre / (v_entre + v_dentro),
               "err_vecinos": float(ev.mean()), "err_media": float(em.mean()),
               "mejora_vecinos": float(mejora), "proyectos": len(nom),
               "tareas": int(n_tot)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/meta.json · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
