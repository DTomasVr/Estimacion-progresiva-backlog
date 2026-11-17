"""E-267 - El usuario da horas de ALGUNAS tareas: ¿cuanto mejora el total?

Es el enfoque que se va a implementar: el usuario escribe las horas de las tareas
que quiera (una, cinco o todas) y el sistema las usa de dos maneras a la vez:

  - esas tareas cuentan por lo que dijo el usuario, no por lo que dice el modelo;
  - de la relacion entre lo que dijo y lo que el modelo predecia para ESAS tareas
    sale un factor que reescala las demas, contraido segun cuantas dio:
        factor = (suma usuario / suma modelo) ^ (m / (m + lambda))
    Con m = 1 solo se ancla esa y el factor apenas se mueve; con mas, pesa mas.

SIMULACION. Cada proyecto del corpus se trata como nuevo: sus predicciones son
fuera de pliegue por proyecto completo (el modelo no lo vio). La entrada del
usuario se simula de dos formas:

  ideal        las horas reales de esas tareas: lo mejor que puede dar un
               usuario, y exactamente lo que pasa cuando la tarea ya se cerro
  estimacion   la estimacion que el usuario dio DE VERDAD (HoursEstimate), en
               los proyectos que la traen en el 80 % de sus tareas o mas. No se
               simula: multiplicar las horas reales por un error independiente
               del tamano da un total inflado que no se parece al de un usuario
               real (se probo: 17 % dandolas todas, frente al 73 % medido)

Se mide sobre (1) el MISMO proyecto y (2) OTRO proyecto del mismo equipo: el
factor aprendido en el proyecto A se aplica en frio al proyecto B de la misma
organizacion, contra el control de aplicarlo a uno de otra organizacion.

Promedio de METRICAS sobre sorteos (que tareas da el usuario es aleatorio).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/entrada_del_usuario")
MIN_TAREAS, SORTEOS = 8, 60
MS = (1, 2, 3, 5, 10, 20)
FRACS = (.25, .50, 1.0)
LAMBDAS = (0, 4, 12, 30)
COBERTURA = .80
ETIQ = ["8-15", "16-40", "41-120", "121-400", "mas de 400"]


def org_de(p):
    if ":" in p:
        return p.split(":")[0]
    return "SiP" if p.startswith("SiP") else p


def carga():
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None),
                      "usu": sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(),
                      "copiada": sw.estimacion_igual_a_real.to_numpy()})
    tam = d.groupby("p").size()
    return d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)


def metricas(tot, real):
    r = np.abs(tot - real) / real
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(tot / real, real / tot) <= 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = carga()
    nom = list(pd.unique(d.p))
    P = len(nom)
    G = [g for _, g in d.groupby("p", sort=False)]
    assert [g.p.iloc[0] for g in G] == nom
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    frio = np.array([x.sum() for x in es_])
    tramo = np.asarray(pd.cut(n, [0, 15, 40, 120, 400, 10 ** 6], labels=ETIQ).astype(str))
    us_ = [g.usu.to_numpy() for g in G]
    con = [np.flatnonzero(~np.isnan(x)) for x in us_]
    sub = np.array([len(c) / len(x) >= COBERTURA for c, x in zip(con, us_)])
    limpio = sub & np.array([g.copiada.mean() < .25 for g in G])
    print(f"{P} proyectos · {len(d):,} tareas · en frio PRED25 {metricas(frio, real)['PRED25']:.3f}", flush=True)
    print(f"con estimacion del usuario: {int(sub.sum())} proyectos (en frio "
          f"{metricas(frio[sub], real[sub])['PRED25']:.3f}) · con menos del 25 % de filas copiadas: "
          f"{int(limpio.sum())} (en frio {metricas(frio[limpio], real[limpio])['PRED25']:.3f})", flush=True)

    def cuantas(j, m):
        return n[j] if m == "todas" else min(int(round(m * n[j])) if isinstance(m, float) else m, n[j])

    def un_sorteo(m, rng):
        """Totales por proyecto para cada (entrada, regla)."""
        tot = {}
        ab = {}
        for j in range(P):
            k = max(1, cuantas(j, m))
            sel = rng.choice(n[j], k, replace=False)
            resto = np.ones(n[j], bool)
            resto[sel] = False
            brazos = [("ideal", re_[j][sel], sel, resto)]
            if sub[j]:
                s2 = rng.choice(con[j], min(k, len(con[j])), replace=False)
                r2 = np.ones(n[j], bool)
                r2[s2] = False
                brazos += [("estimacion", us_[j][s2], s2, r2), ("ideal, mismos proyectos", re_[j][s2], s2, r2)]
            for ent, horas, sel, resto in brazos:
                su, sm, sr = horas.sum(), es_[j][sel].sum(), es_[j][resto].sum()
                tot.setdefault((ent, "solo sustituir"), np.zeros(P))[j] = su + sr
                for lam in LAMBDAS:
                    f = np.exp(np.log(su / sm) * k / (k + lam)) if k + lam else 1.
                    tot.setdefault((ent, f"ancla lambda={lam}"), np.zeros(P))[j] = su + f * sr
                    if lam == 12 and resto.any():
                        ab.setdefault(ent, np.full(P, np.nan))[j] = (
                            abs(f * sr - re_[j][resto].sum()) / re_[j][resto].sum() <= .25)
        return tot, ab

    filas, por_tramo = [], []
    cortes = [(m, f"{m} tarea{'s' if m > 1 else ''}") for m in MS] + \
             [(f, "todas" if f == 1.0 else f"{int(f * 100)} % del backlog") for f in FRACS]
    for m, etiqueta in cortes:
        rng = np.random.default_rng(26700 + int(m * 100))
        acc, acc_ab, acc_tr = {}, {}, {}
        for _ in range(SORTEOS):
            tot, ab = un_sorteo(m, rng)
            for c, t in tot.items():
                msk = np.ones(P, bool) if c[0] == "ideal" else sub
                mm = metricas(t[msk], real[msk])
                if c[0] != "ideal":
                    mm["PRED25_limpios"] = metricas(t[limpio], real[limpio])["PRED25"]
                acc.setdefault(c, []).append(mm)
                if c == ("ideal", "ancla lambda=12"):
                    ok = np.abs(t - real) / real <= .25
                    acc_tr.setdefault(c[0], []).append([ok[tramo == e].mean() for e in ETIQ])
            for e, v in ab.items():
                acc_ab.setdefault(e, []).append(np.nanmean(v if e == "ideal" else v[sub]))
        for (ent, regla), ms in acc.items():
            filas.append(dict(entrada=ent, usuario_da=etiqueta, regla=regla,
                              **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]},
                              PRED25_de_lo_no_dado=(float(np.mean(acc_ab[ent]))
                                                    if regla == "ancla lambda=12" and ent in acc_ab else np.nan)))
        for ent, v in acc_tr.items():
            por_tramo.append(dict(entrada=ent, usuario_da=etiqueta, **dict(zip(ETIQ, np.mean(v, 0)))))
        print(f"  mismo proyecto · {etiqueta} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "mismo_proyecto.csv", index=False)
    pd.DataFrame(por_tramo).to_csv(OUT / "mismo_proyecto_por_tamano.csv", index=False)

    orden = [e for _, e in cortes]
    print(f"\n{'=' * 100}\nMISMO PROYECTO · PRED25 del total (en frio {metricas(frio, real)['PRED25']:.3f})\n{'=' * 100}")
    for ent in ("ideal", "estimacion", "ideal, mismos proyectos"):
        if ent != "ideal":
            print(f"\n(los {int(sub.sum())} proyectos con estimacion; en frio {metricas(frio[sub], real[sub])['PRED25']:.3f})")
        piv = t[t.entrada == ent].pivot(index="usuario_da", columns="regla", values="PRED25").reindex(orden)
        print(f"\nentrada {ent}:")
        print(piv.round(3).to_string())
    print("\nPRED25 solo de las tareas que el usuario NO dio (ancla lambda=12):")
    print(t[t.regla == "ancla lambda=12"].pivot(index="usuario_da", columns="entrada",
                                                values="PRED25_de_lo_no_dado").reindex(orden).round(3).to_string())
    print(f"\nSolo los {int(limpio.sum())} proyectos con pocas filas copiadas · PRED25:")
    print(t[t.entrada == "estimacion"].pivot(index="usuario_da", columns="regla",
                                             values="PRED25_limpios").reindex(orden).round(3).to_string())
    print("\nPor tamano del proyecto (entrada ideal, ancla lambda=12):")
    print(pd.DataFrame(por_tramo).round(3).to_string(index=False))

    # 2 · otro proyecto del mismo equipo
    org = np.array([org_de(p) for p in nom])
    cuenta = pd.Series(org).value_counts()
    con_her = np.flatnonzero(pd.Series(org).map(cuenta).to_numpy() >= 2)
    print(f"\n{len(con_her)} proyectos con hermano en {int((cuenta >= 2).sum())} organizaciones: "
          f"{cuenta[cuenta >= 2].to_dict()}", flush=True)
    filas2 = []
    for m, etiqueta in [(5, "5 tareas"), (10, "10 tareas"), (20, "20 tareas"), (1.0, "todas")]:
        rng = np.random.default_rng(26800 + int(m * 100))
        acc = {}
        for _ in range(SORTEOS):
            th, tc = np.zeros(len(con_her)), np.zeros(len(con_her))
            for a, j in enumerate(con_her):
                her = np.flatnonzero((org == org[j]) & (np.arange(P) != j))
                otr = np.flatnonzero(org != org[j])
                for destino, cand in ((th, her), (tc, otr)):
                    A = rng.choice(cand)
                    k = max(1, cuantas(A, m))
                    sel = rng.choice(n[A], k, replace=False)
                    f = np.exp(np.log(re_[A][sel].sum() / es_[A][sel].sum()) * k / (k + 12))
                    destino[a] = f * frio[j]
            for nombre, tt in (("factor del hermano", th), ("factor de otra organizacion (control)", tc)):
                acc.setdefault(nombre, []).append(metricas(tt, real[con_her]))
        for nombre, ms in acc.items():
            filas2.append(dict(usuario_da_en_A=etiqueta, brazo=nombre,
                               **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}))
        print(f"  otro proyecto · {etiqueta} listo · {time.time() - t0:.0f} s", flush=True)
    filas2.insert(0, dict(usuario_da_en_A="-", brazo="B en frio", **metricas(frio[con_her], real[con_her])))
    t2 = pd.DataFrame(filas2)
    t2.to_csv(OUT / "otro_proyecto_mismo_equipo.csv", index=False)
    print(f"\n{'=' * 100}\nOTRO PROYECTO DEL MISMO EQUIPO ({len(con_her)} proyectos)\n{'=' * 100}")
    print(t2.round(3).to_string(index=False))
    json.dump({"proyectos": P, "sorteos": SORTEOS, "con_estimacion": int(sub.sum()), "limpios": int(limpio.sum()),
               "proyectos_con_hermano": int(len(con_her)),
               "segundos": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
