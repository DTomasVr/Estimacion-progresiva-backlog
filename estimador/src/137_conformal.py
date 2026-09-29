"""E37 - Intervalos con cobertura GARANTIZADA, que es una metrica que si podemos ganar.

E36 deja dos cosas. La buena: las distribuciones aprendidas son mucho mas
nitidas que la climatologia -- intervalo al 80 % de x16 frente a x42 -- o sea que
el modelo si sabe algo sobre la incertidumbre de cada tarea, no solo sobre su
posicion. La mala: estan sobreconfiadas. Cuando dicen 80 % cubren 0,77, y en
proyectos nuevos 0,73.

Eso tiene arreglo, y no uno cualquiera: la prediccion conforme da cobertura
marginal GARANTIZADA sin suponer nada sobre la forma de la distribucion, solo
que las tareas de calibracion y las de prueba son intercambiables. Es la unica
garantia formal que puede dar este proyecto, y ademas es la forma de la salida
que le sirve al producto: "entre 2 y 9 horas" es accionable, "4,3 horas" con un
error tipico del 119 % no lo es.

Tres formas de construir el intervalo, de peor a mejor esperada:

  A  ANCHURA CONSTANTE     el residuo tiene un cuantil global, se suma y se resta
                           al apilado. Cobertura garantizada, misma anchura para
                           todas las tareas. Es la referencia obligada
  B  DEL MODELO, SIN MAS   los cuantiles de la distribucion aprendida tal cual.
                           Adaptativo pero sin garantia, que es lo que fallaba
  C  CONFORME CUANTILICO   los cuantiles del modelo CORREGIDOS por el cuantil de
                           la no conformidad en validacion (Romano, Patterson y
                           Candes, 2019). Adaptativo Y garantizado

La pregunta que decide si el modelo aporta algo sobre la incertidumbre no es la
cobertura -- A y C la cumplen las dos por construccion -- sino la ANCHURA a
igual cobertura, y sobre todo la cobertura CONDICIONAL: si el intervalo de A
cubre de mas a las tareas pequenas y de menos a las grandes, y el de C no, el
modelo esta aportando algo real aunque la cobertura marginal empate.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/conformal")
S = "=" * 124
NIVELES = (.5, .8, .9)


def cuantil_discreto(sop, P, q):
    o = np.argsort(sop, 1)
    s = np.take_along_axis(sop, o, 1)
    p = np.take_along_axis(P, o, 1)
    return np.take_along_axis(s, (p.cumsum(1) >= q).argmax(1)[:, None], 1).ravel()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    va = (sw.split_t3 == "val").to_numpy()
    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    E5 = np.load(CACHE / "e5-large.npy")
    AJ = np.load(CACHE / "e5-ajustado.npy")
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Z = np.column_stack([np.mean([m.predict(np.hstack([E5, SV])) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Xs), f["svr"].predict(Xs),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)
    # el ultimo cuarto del entrenamiento por orden cronologico, para calibrar con
    # lo mas parecido en el tiempo a lo que viene despues
    tr = (sw.split_t3 == "train").to_numpy()
    o = sw.orden.to_numpy(float)
    corte_t = np.nanquantile(o[tr], .75)
    tardio = tr & (o >= corte_t)
    print(f"calibracion tardia: {int(tardio.sum()):,} tareas del ultimo cuarto del "
          f"entrenamiento · validacion {int(va.sum()):,}")
    Pabs = np.load("results/distribucional/P.npy").astype(float)
    val = np.load("results/distribucional/val.npy")
    sop = np.repeat(np.log(val)[None, :], len(sw), 0)

    filas, cond = [], []
    for niv in NIVELES:
        a = (1 - niv) / 2
        def corte(v, m):
            n = int(m.sum())
            return float(np.sort(v[m])[min(int(np.ceil((n + 1) * niv)), n) - 1])

        # A - anchura constante, conforme sobre el residuo absoluto
        w = corte(np.abs(ly - lp), va)
        A = (lp - w, lp + w)
        # B - cuantiles del modelo, sin correccion
        qlo, qhi = cuantil_discreto(sop, Pabs, a), cuantil_discreto(sop, Pabs, 1 - a)
        B = (qlo, qhi)
        # C - conforme cuantilico: E = max(qlo - y, y - qhi)
        e = np.maximum(qlo - ly, ly - qhi)
        c = corte(e, va)
        C = (qlo - c, qhi + c)
        # D - lo mismo, pero calibrado con el tramo de entrenamiento MAS CERCANO
        # en el tiempo al bloque de prueba. La garantia conforme pide que
        # calibracion y prueba sean intercambiables, y con una particion
        # cronologica no lo son; esta es la version practica del remedio.
        cd = corte(e, tardio)
        D = (qlo - cd, qhi + cd)
        for etq, (lo, hi) in (("A · anchura constante (conforme)", A),
                              ("B · cuantiles del modelo, sin corregir", B),
                              ("C · conforme cuantilico (adaptativo)", C),
                              ("D · C calibrado con lo mas reciente", D)):
            filas.append(dict(nivel=f"{niv:.0%}", metodo=etq, bloque="validacion (calibracion)",
                              cobertura=float(((ly[va] >= lo[va]) & (ly[va] <= hi[va])).mean()),
                              anchura_x=float(np.exp(np.median(hi[va] - lo[va]))),
                              anchura_media_x=float(np.exp((hi[va] - lo[va]).mean()))))
            for b, m in bl.items():
                dentro = (ly[m] >= lo[m]) & (ly[m] <= hi[m])
                filas.append(dict(nivel=f"{niv:.0%}", metodo=etq, bloque=b,
                                  cobertura=float(dentro.mean()),
                                  anchura_x=float(np.exp(np.median(hi[m] - lo[m]))),
                                  anchura_media_x=float(np.exp((hi[m] - lo[m]).mean()))))
            # cobertura condicional por tramo del tamano predicho, en la prueba fija
            if niv == .8:
                m = bl["prueba fija"]
                tramo = pd.qcut(lp[m], 4, labels=["muy pequena", "pequena", "mediana", "grande"])
                d = pd.DataFrame({"t": tramo,
                                  "cub": (ly[m] >= lo[m]) & (ly[m] <= hi[m]),
                                  "anch": np.exp(hi[m] - lo[m])})
                for t, g in d.groupby("t", observed=True):
                    cond.append(dict(metodo=etq, tramo=str(t), n=len(g),
                                     cobertura=float(g.cub.mean()),
                                     anchura_x=float(g.anch.median())))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(cond)
    for niv in NIVELES:
        print(f"\n{S}\nINTERVALOS AL {niv:.0%}\n{S}")
        v = t[t.nivel == f"{niv:.0%}"]
        print(v.pivot(index="metodo", columns="bloque",
                      values=["cobertura", "anchura_x"]).round(3).to_string())
    print(f"\n{S}\nCOBERTURA CONDICIONAL AL 80 % POR TAMANO PREDICHO (prueba fija)\n{S}")
    print(c.pivot(index="metodo", columns="tramo",
                  values=["cobertura", "anchura_x"]).round(3).to_string())
    print("\n  Lectura: la fila de validacion es la cobertura EN LA PROPIA CALIBRACION,")
    print("  que por construccion sale en el nivel pedido. Lo que interese comparar es")
    print("  cuanto de eso sobrevive en los bloques de prueba: si cae, la garantia")
    print("  conforme se esta rompiendo, y la unica forma de que se rompa es que la")
    print("  calibracion y la prueba no sean intercambiables. Con una particion")
    print("  cronologica es exactamente lo que cabe esperar.")
    t.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "condicional.csv", index=False)
    json.dump({"niveles": list(NIVELES)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv y condicional.csv")


if __name__ == "__main__":
    main()
