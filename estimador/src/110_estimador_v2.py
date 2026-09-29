"""El estimador v2: todo lo que sobrevivio a la noche del 29/09, en un objeto.

Cambia respecto a v1 en cuatro cosas, todas medidas:

  CODIFICADOR   e5-large en vez de mpnet, mas tf-idf de caracteres y de
                palabras. Prueba fija: R² 0,109 -> 0,152.
  BOLSA         siete HGB con submuestreo, reentrenados con train + val.
                R² 0,152 -> 0,164.
  ANCLAS        el usuario da las horas de k tareas del backlog y el
                desplazamiento del proyecto se estima con contraccion
                James-Stein (lambda = 12, elegido en validacion). Sin
                contraccion, anclar con una tarea EMPEORA. En proyectos nuevos
                con 20 anclas: R² 0,056 -> 0,267.
  CORRECTOR     si el usuario estima una tarea, se corrige su numero en vez de
                ignorarlo. En proyectos nuevos: R² 0,056 -> 0,302; con anclas
                ademas, 0,424.

Se conserva de v1 lo que seguia valiendo: smearing de Duan para los totales,
factor de estilo para el texto del generador, bandas conformes y abstencion por
riesgo aprendido.

Nota de calibracion: la bolsa desplegada se entrena con train + val, pero las
bandas, el umbral de abstencion y el corrector se calibran con la version
entrenada SOLO con train, leida sobre val. Es ligeramente conservador y es la
unica forma de no calibrar sobre datos vistos.

    from importlib.util import spec_from_file_location, module_from_spec
    s = spec_from_file_location("e", "src/110_estimador_v2.py")
    m = module_from_spec(s); s.loader.exec_module(m)
    est = m.EstimadorV2()
    r = est.estimar(["Store payment data in the member profile"],
                    proyecto=["p1"], estimacion_usuario=[None])
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 225)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
BASE_COD = "intfloat/multilingual-e5-large"
PAQUETE = Path("models/estimador_agix_v2.joblib")
OUT = Path("results/estimador_v2")
LAMBDA, N_BOLSA, COBERTURA = 12.0, 7, 0.5
SEP = "=" * 128


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def hgb(semilla=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=semilla, **kw)


class EstimadorV2:
    def __init__(self, paquete=PAQUETE, dispositivo=None):
        self.p = joblib.load(paquete)
        self._st, self._dev = None, dispositivo

    @property
    def codificador(self):
        if self._st is None:
            import torch
            from sentence_transformers import SentenceTransformer
            d = self._dev or ("cuda" if torch.cuda.is_available() else "cpu")
            self._st = SentenceTransformer(self.p["codificador"], device=d)
        return self._st

    def vectorizar(self, textos):
        V = self.codificador.encode(["query: " + t for t in textos],
                                    normalize_embeddings=False, show_progress_bar=False)
        partes = [V.astype(np.float32)]
        for tf, sv in self.p["tfidf"]:
            partes.append(sv.transform(tf.transform(textos)).astype(np.float32))
        return np.hstack(partes)

    def estimar(self, textos, proyecto=None, estimacion_usuario=None,
                anclas=None, estilo_generador=True, umbral="absoluto"):
        """Una fila por tarea.

        proyecto            identificador para agrupar el lote; si falta, todo
                            el lote se trata como un proyecto
        estimacion_usuario  horas que cree el usuario, o None por tarea
        anclas              horas REALES conocidas de algunas tareas (None si no)
        """
        n = len(textos)
        X = self.vectorizar(list(textos))
        lp = np.mean([m.predict(X) for m in self.p["bolsa"]], 0)
        if estilo_generador:
            lp = lp + np.log(self.p["factor_estilo"])
        proy = np.array(proyecto if proyecto is not None else ["_"] * n, dtype=object)

        if anclas is not None:
            anc = np.array([np.nan if a is None else float(a) for a in anclas])
            for p in np.unique(proy):
                k = (proy == p) & ~np.isnan(anc)
                if k.sum():
                    bruto = float(np.median(np.log(anc[k]) - lp[k]))
                    lp[proy == p] += bruto * k.sum() / (k.sum() + self.p["lambda"])
        horas = np.exp(lp)

        if estimacion_usuario is not None:
            est = np.array([np.nan if e is None else float(e) for e in estimacion_usuario])
            hay = ~np.isnan(est)
            if hay.any():
                Z = np.column_stack([np.log(np.where(hay, est, 1)), lp])
                horas = np.where(hay, np.where(hay, est, 1) *
                                 np.exp(self.p["corrector"].predict(Z)), horas)
        k80 = self.p["factor_banda80"]
        anchura = self.p["q90"].predict(X) - self.p["q10"].predict(X)
        conf = self.p["clasificador"].predict_proba(X).max(1)
        Xn = X[:, :self.p["dim_emb"]]
        cerca = ((Xn / np.linalg.norm(Xn, axis=1, keepdims=True)) @ self.p["centros"].T).max(1)
        riesgo = self.p["riesgo"].predict(np.column_stack([anchura, conf, cerca]))
        responder = (riesgo <= self.p["umbral_riesgo"] if umbral == "absoluto"
                     else riesgo <= np.quantile(riesgo, self.p["cobertura"]))
        return pd.DataFrame({"texto": list(textos), "proyecto": proy, "horas": horas,
                             "banda_baja": horas / k80, "banda_alta": horas * k80,
                             "riesgo": riesgo, "responder": responder})

    def total_proyecto(self, r):
        s = self.p["smearing"]
        total = float(r.horas.sum() * s)
        return {"tareas": len(r), "suma_de_medianas_h": float(r.horas.sum()),
                "total_esperado_h": total, "total_esperado_dias": total / 8,
                "total_ajustado_h": total / self.p["suma_residual"],
                "sin_respuesta": int((~r.responder).sum())}


def construir():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()
    va_m = (sw.split_t3 == "val").to_numpy()
    trva = tr_m | va_m

    Xe = np.load(CACHE / "e5-large.npy")
    tfidf, partes = [], [Xe]
    for kw in (dict(analyzer="char_wb", ngram_range=(3, 5)), dict(ngram_range=(1, 2))):
        tf = TfidfVectorizer(min_df=3, max_features=200_000, sublinear_tf=True,
                             **kw).fit(tit[tr_m])
        M = tf.transform(tit)
        sv = TruncatedSVD(256, random_state=0).fit(M[tr_m])
        tfidf.append((tf, sv))
        partes.append(sv.transform(M).astype(np.float32))
    X = np.hstack(partes)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    print(f"vista {X.shape} · train {tr_m.sum():,} · val {va_m.sum():,}")

    # modelo de calibracion (solo train) y modelo desplegado (train + val)
    solo_tr = hgb().fit(X[tr_m], ly[tr_m])
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(trva)
    bolsa = []
    for s in range(N_BOLSA):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(hgb(s, max_features=.8).fit(X[sub], ly[sub]))
        print(f"  bolsa {s + 1}/{N_BOLSA}", flush=True)

    q10 = hgb(loss="quantile", quantile=.1).fit(X[tr_m], ly[tr_m])
    q90 = hgb(loss="quantile", quantile=.9).fit(X[tr_m], ly[tr_m])
    tramos = np.asarray(pd.cut(y, [0, 1, 4, 16, np.inf],
                               labels=["<1 h", "1-4 h", "4-16 h", ">16 h"], right=False))
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0).fit(
        X[tr_m], tramos[tr_m])
    print("  cuantiles y clasificador de tramo ajustados")

    lp_va = solo_tr.predict(X[va_m])
    res = np.abs(ly[va_m] - lp_va)
    banda80 = float(np.exp(np.quantile(res, .8)))
    Xn = Xe / np.linalg.norm(Xe, axis=1, keepdims=True)
    centros = Xn[tr_m].astype(np.float32)
    senales = np.column_stack([q90.predict(X[va_m]) - q10.predict(X[va_m]),
                               clf.predict_proba(X[va_m]).max(1),
                               (Xn[va_m] @ centros.T).max(1)])
    riesgo = Ridge(alpha=1.0).fit(senales, res)
    umbral = float(np.quantile(riesgo.predict(senales), COBERTURA))

    # corrector de la estimacion del usuario (C2), con frio fuera de pliegue
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    ptr = pt[pt.split_t3 == "train"]
    pend, corte = np.polyfit(np.log(ptr.story_point.to_numpy(float)),
                             np.log(ptr.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    hay = sw.est.notna().to_numpy()
    oof = np.mean([m.predict(X) for m in bolsa], 0)
    it = np.flatnonzero(tr_m)
    for a, b in KFold(5, shuffle=True, random_state=0).split(it):
        oof[it[b]] = hgb().fit(X[it[a]], ly[it[a]]).predict(X[it[b]])
    k = tr_m & hay
    Z = np.column_stack([np.log(np.where(hay, sw.est, 1).astype(float)), oof])
    corrector = hgb().fit(Z[k], (ly - np.log(sw.est.fillna(1).to_numpy(float)))[k])

    resid = ly[tr_m] - solo_tr.predict(X[tr_m])
    S = float(np.mean(np.exp(resid)))
    suma_res = float((np.exp(lp_va) * S).sum() / y[va_m].sum())
    corr = json.load(open("results/correcciones.json", encoding="utf-8"))

    PAQUETE.parent.mkdir(exist_ok=True)
    joblib.dump({"bolsa": bolsa, "tfidf": tfidf, "codificador": BASE_COD,
                 "dim_emb": Xe.shape[1], "q10": q10, "q90": q90, "clasificador": clf,
                 "riesgo": riesgo, "umbral_riesgo": umbral, "cobertura": COBERTURA,
                 "centros": centros, "corrector": corrector, "lambda": LAMBDA,
                 "smearing": S, "suma_residual": suma_res,
                 "factor_estilo": corr["estilo_medio"], "factor_banda80": banda80,
                 "corpus": "v4 sin Apache", "n_train": int(trva.sum()),
                 "idioma_esperado": "ingles", "vista": "titulo",
                 "sha_datos": E.data_fingerprint(DATA),
                 "fecha": time.strftime("%Y-%m-%d")}, PAQUETE, compress=3)
    print(f"escrito {PAQUETE} ({PAQUETE.stat().st_size / 1e6:.0f} MB) "
          f"en {time.time() - t0:.0f} s")
    print(f"  banda 80 % ×{banda80:.2f} · smearing {S:.3f} · "
          f"suma residual {suma_res:.3f} · umbral {umbral:.4f}")
    return sw, X, y, ly


def verificar(sw, X, y, ly):
    est = EstimadorV2()
    p = est.p
    lp = np.mean([m.predict(X) for m in p["bolsa"]], 0)
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    anchura = p["q90"].predict(X) - p["q10"].predict(X)
    conf = p["clasificador"].predict_proba(X).max(1)
    Xn = X[:, :p["dim_emb"]]
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    cerca = (Xn @ p["centros"].T).max(1)
    riesgo = p["riesgo"].predict(np.column_stack([anchura, conf, cerca]))
    filas = []
    for b, k in bloques.items():
        yb, pb = y[k], np.exp(lp[k])
        resp = riesgo[k] <= p["umbral_riesgo"]
        filas.append(dict(bloque=b, subconjunto="todas", n=int(k.sum()),
                          cobertura=1.0, **metricas(yb, pb)))
        if resp.sum() > 20:
            filas.append(dict(bloque=b, subconjunto="respondidas", n=int(resp.sum()),
                              cobertura=float(resp.mean()),
                              **metricas(yb[resp], pb[resp])))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nVERIFICACIÓN DEL v2 SOBRE LOS BLOQUES CONGELADOS\n{SEP}")
    print(t.round(4).to_string(index=False))
    Path(OUT).mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "verificacion.csv", index=False)
    print(f"\nescrito {OUT}/verificacion.csv")


if __name__ == "__main__":
    sw, X, y, ly = construir()
    verificar(sw, X, y, ly)
