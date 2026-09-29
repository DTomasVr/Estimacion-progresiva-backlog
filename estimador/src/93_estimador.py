"""El estimador completo: un solo objeto con las tres piezas que sobrevivieron.

De las cinco intervenciones medidas en 89-92, tres se integran y dos quedan
descartadas con su medicion:

  SMEARING (paso 1)   para el TOTAL del proyecto, no para la tarea. Sin el, la
                      suma de un backlog se queda un 60-66 % corta, porque
                      exponenciar la prediccion en log devuelve la mediana y lo
                      que se suma es la media.
  ESTILO (paso 2)     el texto del generador infla las predicciones; x0,831
                      ajustado en un bloque y verificado en el otro lo corrige.
  ABSTENCION (paso 5) el modelo sabe en parte donde va a fallar. Al responder
                      solo la mitad mas segura, la proporcion de tareas de mas
                      de 16 h entre las respondidas cae de 0,091 a 0,041: lo que
                      se calla son justo las que rompen una planificacion.

  bandas condicionales (paso 3) y clasificador de tramo (paso 4) NO se integran:
  ninguna arregla la cobertura por tramo ni encuentra las tareas grandes. La
  medicion esta en results/bandas_condicionales.csv y clasificador_tramo.csv.

El riesgo no se toma de una sola senal: se aprende en la validacion un modelo
que predice el error logarithmico a partir de la anchura de la banda, la
confianza del clasificador y la cercania al entrenamiento. Es lo que la
literatura de regresion selectiva llama un estimador de riesgo, y se verifica
contra abstenerse al azar antes de creerselo.

Uso:
    from importlib.util import spec_from_file_location, module_from_spec
    s = spec_from_file_location("est", "src/93_estimador.py")
    m = module_from_spec(s); s.loader.exec_module(m)
    e = m.Estimador()
    r = e.estimar(["Store payment data in the member profile"], estilo_generador=True)
    e.total_proyecto(r)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
PAQUETE = Path("models/estimador_agix.joblib")
COBERTURA_OBJETIVO = 0.5
SEP = "=" * 126


class Estimador:
    """Todo el componente de estimacion en un objeto."""

    def __init__(self, paquete=PAQUETE, dispositivo=None):
        p = joblib.load(paquete)
        self.p = p
        self._st = None
        self._dispositivo = dispositivo

    @property
    def codificador(self):
        if self._st is None:
            import torch
            from sentence_transformers import SentenceTransformer
            d = self._dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
            self._st = SentenceTransformer(self.p["codificador"], device=d)
        return self._st

    def _rasgos(self, X):
        """Solo el texto. Medido en 85 y confirmado aqui: con la categoria real
        el modelo es mejor (0,6867 frente a 0,6971), pero en produccion la
        categoria no llega y poner "task" por defecto lo empeora hasta 0,7326.
        Sin categoria es la unica configuracion honesta."""
        return X

    def estimar(self, textos, estilo_generador=True, vectores=None,
                umbral="absoluto"):
        """Una fila por tarea: horas, banda, riesgo y si conviene responder.

        `textos` en INGLES y en formato de titulo. `estilo_generador=True`
        aplica el factor medido en 87 para texto escrito por el generador.

        `umbral`: "absoluto" usa el corte calibrado en validacion -- honesto,
        pero sobre texto del generador, que esta fuera de la variedad de
        entrenamiento, deja sin responder el 83 %. "relativo" responde la mitad
        mas segura DE ESTE LOTE, que es lo que un producto puede enseñar, a
        cambio de no garantizar el nivel de error.
        """
        X = vectores if vectores is not None else self.codificador.encode(
            list(textos), normalize_embeddings=False, show_progress_bar=False)
        R = self._rasgos(X)
        lp = self.p["regresor"].predict(R)
        horas = np.exp(lp)
        if estilo_generador:
            horas = horas * self.p["factor_estilo"]
        k = self.p["factor_banda80"]
        anchura = self.p["q90"].predict(R) - self.p["q10"].predict(R)
        prob = self.p["clasificador"].predict_proba(R)
        confianza = prob.max(1)
        cercania = (X / np.linalg.norm(X, axis=1, keepdims=True)) @ self.p["centros"].T
        cercania = cercania.max(1)
        riesgo = self.p["riesgo"].predict(np.column_stack([anchura, confianza, cercania]))
        tramo = self.p["clasificador"].classes_[prob.argmax(1)]
        return pd.DataFrame({
            "texto": list(textos) if vectores is None else np.arange(len(X)),
            "horas": horas, "banda_baja": horas / k, "banda_alta": horas * k,
            "tramo_probable": tramo, "riesgo": riesgo,
            "responder": (riesgo <= self.p["umbral_riesgo"] if umbral == "absoluto"
                          else riesgo <= np.quantile(riesgo, self.p["cobertura_objetivo"])),
        })

    def total_proyecto(self, r):
        """El total no es la suma de las medianas: hay que deshacer la retransformacion."""
        s = self.p["smearing"]
        total = float(r.horas.sum() * s)
        return {"tareas": len(r),
                "suma_de_medianas_h": float(r.horas.sum()),
                "total_esperado_h": total,
                "total_esperado_dias": total / 8,
                "correccion_infraestimacion": self.p["suma_residual"],
                "total_ajustado_h": total / self.p["suma_residual"],
                "tareas_sin_respuesta": int((~r.responder).sum())}


# ------------------------------------------------------------------ construccion
def construir():
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    a = joblib.load("models/candidato_titulos.joblib")
    corr = json.load(open("results/correcciones.json", encoding="utf-8"))

    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    oh = None                       # solo texto, ver _rasgos

    def rasgos(d):
        return emb[d.fe.to_numpy()]

    tr = sw[(sw.split_t3 == "train") & (~sw.apache)]
    va = sw[(sw.split_t3 == "val") & (~sw.apache)]

    def hgb(**kw):
        return dict(max_iter=300, max_depth=6, learning_rate=.06,
                    l2_regularization=1., random_state=0, **kw)

    Xtr_r, ytr = rasgos(tr), np.log(tr.effort_hours.to_numpy())
    reg = HistGradientBoostingRegressor(**hgb()).fit(Xtr_r, ytr)
    cuant = {q: HistGradientBoostingRegressor(**hgb(loss="quantile", quantile=q)).fit(
        Xtr_r, ytr) for q in (.1, .9)}
    clf = HistGradientBoostingClassifier(**hgb()).fit(
        Xtr_r, np.asarray(pd.cut(tr.effort_hours, [0, 1, 4, 16, np.inf],
                                 labels=["<1 h", "1-4 h", "4-16 h", ">16 h"], right=False)))
    a = dict(a, regresor=reg)
    cl = {"clasificador": clf}
    print("reajustados regresor, cuantiles y clasificador sobre solo texto")

    # el riesgo se aprende en validacion, nunca en prueba
    Rva = rasgos(va)
    lp = a["regresor"].predict(Rva)
    err = np.abs(np.log(va.effort_hours.to_numpy()) - lp)
    Xva = emb[va.fe.to_numpy()]
    Xn = Xva / np.linalg.norm(Xva, axis=1, keepdims=True)
    Xtr = emb[tr.fe.to_numpy()]
    Xtrn = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    senales = np.column_stack([cuant[.9].predict(Rva) - cuant[.1].predict(Rva),
                               cl["clasificador"].predict_proba(Rva).max(1),
                               (Xn @ Xtrn.T).max(1)])
    riesgo = Ridge(alpha=1.0).fit(senales, err)
    umbral = float(np.quantile(riesgo.predict(senales), COBERTURA_OBJETIVO))
    print(f"estimador de riesgo ajustado en {len(va):,} filas de validación · "
          f"umbral al {COBERTURA_OBJETIVO:.0%} de cobertura = {umbral:.4f}")
    print(f"  pesos: anchura {riesgo.coef_[0]:+.4f} · confianza {riesgo.coef_[1]:+.4f} "
          f"· cercanía {riesgo.coef_[2]:+.4f}")

    # cuanto se queda corta la suma, ya con smearing, para poder decirlo
    res = np.log(tr.effort_hours.to_numpy()) - a["regresor"].predict(rasgos(tr))
    S = float(np.mean(np.exp(res)))
    pv = np.exp(lp) * S
    suma_residual = float(pv.sum() / va.effort_hours.sum())
    print(f"  smearing S = {S:.4f} · con él la suma en validación sigue siendo el "
          f"{suma_residual:.1%} de la real")

    joblib.dump({
        "regresor": a["regresor"], "onehot": oh, "codificador": a["codificador"],
        "q10": cuant[.1], "q90": cuant[.9], "clasificador": cl["clasificador"],
        "riesgo": riesgo, "umbral_riesgo": umbral, "cobertura_objetivo": COBERTURA_OBJETIVO,
        "centros": Xtrn.astype(np.float32), "smearing": S,
        "factor_estilo": corr["estilo_medio"], "factor_banda80": a["bandas"][
            "validación sin Apache"]["factor80"], "suma_residual": suma_residual,
        "categoria_por_defecto": "task", "idioma_esperado": "ingles", "vista": "titulo",
        "corpus": a["corpus"], "n_train": a["n_train"], "sha_datos": a["sha_datos"],
        "notas": "smearing solo para totales; factor de estilo solo si el texto viene "
                 "del generador; abstencion con umbral al 50 % de cobertura",
    }, PAQUETE, compress=3)
    print(f"escrito {PAQUETE} ({PAQUETE.stat().st_size / 1e6:.1f} MB)")
    return sw, emb


def verificar(sw, emb):
    e = Estimador()
    p = e.p
    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}
    filas = []
    for b, d in bloques.items():
        X = emb[d.fe.to_numpy()]
        r = e.estimar(d.title.astype(str).tolist(), estilo_generador=False, vectores=X)
        y = d.effort_hours.to_numpy()
        for etq, k in (("todas", np.ones(len(d), bool)),
                       ("solo las respondidas", r.responder.to_numpy())):
            h = r.horas.to_numpy()[k]
            m = E.point_metrics(y[k], np.clip(h, .01, None))
            cub = ((y[k] >= r.banda_baja.to_numpy()[k]) &
                   (y[k] <= r.banda_alta.to_numpy()[k])).mean()
            filas.append(dict(bloque=b, subconjunto=etq, n=int(k.sum()),
                              cobertura=float(k.mean()), MdMRE=m["MdMRE"],
                              MdAElog=float(np.median(np.abs(np.log(
                                  np.clip(h, .01, None) / y[k])))),
                              PRED25=m["PRED25"], banda80=float(cub),
                              frac_grandes=float((y[k] > 16).mean()),
                              suma_con_smearing=float(h.sum() * p["smearing"] / y[k].sum())))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nVERIFICACIÓN SOBRE LOS BLOQUES CONGELADOS\n{SEP}")
    print(t.round(4).to_string(index=False))
    t.to_csv("results/estimador_verificacion.csv", index=False)
    return e


def demostrar(e):
    gen = Path("F:/Software Task generation Model/salida_gen.json")
    if not gen.exists():
        return
    d = json.load(open(gen, encoding="utf-8"))
    tex, dom = [], []
    for k, v in d.items():
        for h in v["historias"]:
            es, en = h.get("tareas_es") or [], h.get("tareas") or []
            for i, t in enumerate(es):
                if i < len(en):
                    tex.append(en[i])
                    dom.append(k)
    r = e.estimar(tex, estilo_generador=True)
    r["dominio"] = dom
    print(f"\n{SEP}\nLO QUE MOSTRARÍA AGIX PARA UN BACKLOG GENERADO\n{SEP}")
    for modo in ("absoluto", "relativo"):
        r = e.estimar(tex, estilo_generador=True, umbral=modo)
        r["dominio"] = dom
        print(f"\n  umbral {modo}: responde el {r.responder.mean():.0%} de las tareas")
        for _, x in pd.concat([r[r.responder].head(3), r[~r.responder].head(3)]).iterrows():
            est = (f"{x.horas:5.2f} h  [{x.banda_baja:4.1f}–{x.banda_alta:5.1f}]"
                   if x.responder else "  — preguntar al usuario —      ")
            print(f"    {est}  {x.tramo_probable:7s} {x.texto[:60]}")
    tot = e.total_proyecto(r)
    print(f"\n  {tot['tareas']} tareas · {tot['tareas_sin_respuesta']} sin respuesta "
          f"({tot['tareas_sin_respuesta'] / tot['tareas']:.0%})")
    print(f"  suma de las medianas      {tot['suma_de_medianas_h']:6.0f} h")
    print(f"  total esperado (smearing) {tot['total_esperado_h']:6.0f} h "
          f"({tot['total_esperado_dias']:.0f} días-persona)")
    print(f"  ajustado por lo que la suma se sigue quedando corta "
          f"(×{1 / tot['correccion_infraestimacion']:.2f}): "
          f"{tot['total_ajustado_h']:6.0f} h")
    r.to_csv("results/estimador_demo.csv", index=False)


if __name__ == "__main__":
    sw, emb = construir()
    e = verificar(sw, emb)
    demostrar(e)
    print("\nescrito results/estimador_verificacion.csv y estimador_demo.csv")
