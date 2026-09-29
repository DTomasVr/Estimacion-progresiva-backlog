"""El estimador v4: el v3 mas el canal de analogia, que es lo unico que E30-E41 mejoro.

E41 deja un resultado que hay que llevar al artefacto: la moda sobre la rejilla de
valores frecuentes de las 25 tareas de entrenamiento con el titulo mas parecido
sube PRED(25) y factor 2 A LA VEZ, con significancia en dos de los tres bloques,
que es algo que no consiguio ningun otro brazo de toda la serie.

Pero no lo sube gratis: cuesta R2 (0,199 -> 0,072) y MdAElog (0,783 -> 0,847). No
hay un numero que gane las dos cosas, asi que el artefacto no debe elegir por el
usuario: debe dar los dos y decir para que sirve cada uno.

    horas             la del apilado. Es la que minimiza el error y la que hay que
                      sumar para estimar un proyecto entero
    valor_probable    la de la analogia. Es la que mas veces cae dentro del +-25 %,
                      y la que tiene sentido ensenarle a una persona, porque cae
                      sobre valores que de verdad se observan (1 h, 2 h, 4 h) en
                      vez de sobre 1,83 h
    acuerdo           si los vecinos concuerdan entre si. Cuando no concuerdan, el
                      valor_probable no vale y hay que quedarse con horas

Todo lo demas del v3 sigue igual: anclas con contraccion, corrector de la
estimacion del usuario, familiaridad, banda conforme al 80 %, smearing para el
total y factor de estilo del generador.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
V3 = Path("models/estimador_agix_v3.joblib")
V4 = Path("models/estimador_agix_v4.joblib")
OUT = Path("results/estimador_v4")
S = "=" * 120
K_VEC = 25
NIVELES = [(.97, "ya hiciste algo casi igual"), (.93, "algo parecido"),
           (.85, "de la misma familia"), (-2, "esto es nuevo")]


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


class EstimadorV4:
    """Igual que el v3 en la interfaz; anade valor_probable y acuerdo."""

    def __init__(self, paquete=V4, dispositivo=None):
        self.p = joblib.load(paquete)
        self._st, self._aj, self._dev = None, None, dispositivo

    @property
    def codificador(self):
        if self._st is None:
            import torch
            from sentence_transformers import SentenceTransformer
            d = self._dev or ("cuda" if torch.cuda.is_available() else "cpu")
            self._st = SentenceTransformer(self.p["codificador"], device=d)
        return self._st

    @property
    def codificador_ajustado(self):
        if self._aj is None:
            import torch
            from sentence_transformers import SentenceTransformer, models
            d = self._dev or ("cuda" if torch.cuda.is_available() else "cpu")
            w = models.Transformer(self.p["codificador_ajustado"], max_seq_length=48)
            self._aj = SentenceTransformer(
                modules=[w, models.Pooling(w.get_word_embedding_dimension(), "mean")],
                device=d)
        return self._aj

    def _vistas(self, textos):
        V = self.codificador.encode(["query: " + t for t in textos],
                                    normalize_embeddings=False,
                                    show_progress_bar=False).astype(np.float32)
        Ms = sparse.hstack([self.p["tfc"].transform(textos),
                            self.p["tfw"].transform(textos)]).tocsr()
        Xd = np.hstack([V] + [sv.transform(tf.transform(textos)).astype(np.float32)
                              for tf, sv in zip((self.p["tfc"], self.p["tfw"]),
                                                self.p["svds"])])
        return V, Xd, Ms

    def analogia(self, V):
        """Moda sobre la rejilla de los K vecinos mas parecidos del entrenamiento."""
        Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        A = self.p["vec_V"].astype(np.float32)
        yv = self.p["vec_ly"]
        rej = self.p["rejilla"]
        s = Vn @ A.T
        j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
        v = yv[j]
        med = np.median(v, 1)
        moda = rej[np.abs(np.log(rej)[None, :] - med[:, None]).argmin(1)]
        return moda, v.std(1)

    def estimar(self, textos, proyecto=None, estimacion_usuario=None, anclas=None,
                historial=None, estilo_generador=True):
        textos = list(textos)
        n = len(textos)
        V, Xd, Ms = self._vistas(textos)
        aj = self.codificador_ajustado.encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        Z = np.column_stack([np.mean([m.predict(Xd) for m in self.p["bolsa"]], 0),
                             self.p["ridge"].predict(Ms), self.p["svr"].predict(Ms),
                             self.p["hgb_ajustado"].predict(aj)])
        lp = self.p["apilado"].predict(Z)
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
                Zc = np.hstack([V, np.log(np.where(hay, est, 1))[:, None], lp[:, None]])
                horas = np.where(hay, np.where(hay, est, 1) *
                                 np.exp(self.p["corrector"].predict(Zc)), horas)
        prob, disp = self.analogia(V)
        acuerdo = disp <= self.p["umbral_dispersion"]
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        if historial:
            H = self.codificador.encode(["query: " + t for t in historial],
                                        normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)
            sim = (Vn @ H.T).max(1)
        else:
            sim = (Vn @ self.p["centros"].T).max(1)
        nivel = np.array([next(e for u, e in NIVELES if s >= u) for s in sim], dtype=object)
        k80 = self.p["factor_banda80"]
        return pd.DataFrame({"texto": textos, "proyecto": proy, "horas": horas,
                             "valor_probable": np.where(acuerdo, prob, horas),
                             "acuerdo": acuerdo, "dispersion_vecinos": disp,
                             "banda_baja": horas / k80, "banda_alta": horas * k80,
                             "familiaridad": sim, "nivel": nivel})


def construir():
    t0 = time.time()
    p = joblib.load(V3)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    tr = (sw.split_t3 == "train").to_numpy()
    va = (sw.split_t3 == "val").to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    c = pd.Series(y[tr]).value_counts()
    rej = np.sort(c[c >= 60].index.to_numpy())
    p["vec_V"] = Vn[tr].astype(np.float16)
    p["vec_ly"] = ly[tr].astype(np.float32)
    p["rejilla"] = rej
    p["k_vecinos"] = K_VEC

    # umbral de acuerdo, elegido en validacion igual que en E41
    A = Vn[tr].astype(np.float32)
    s = Vn @ A.T
    j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
    v = ly[tr][j]
    moda = rej[np.abs(np.log(rej)[None, :] - np.median(v, 1)[:, None]).argmin(1)]
    disp = v.std(1)
    tit = sw.title.fillna("").astype(str)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((p["tfc"], p["tfw"]), p["svds"])])
    Xs = sparse.hstack([p["tfc"].transform(tit), p["tfw"].transform(tit)]).tocsr()
    lp = p["apilado"].predict(np.column_stack([
        np.mean([m.predict(np.hstack([E5, SV])) for m in p["bolsa"]], 0),
        p["ridge"].predict(Xs), p["svr"].predict(Xs),
        p["hgb_ajustado"].predict(np.load(CACHE / "e5-ajustado.npy"))]))
    mejor_u, mejor_v = None, -1
    for u in np.quantile(disp[va], np.arange(.05, 1., .05)):
        q = np.where(disp <= u, moda, np.exp(lp))
        val = metricas(y[va], q[va])["PRED25"]
        if val > mejor_v:
            mejor_u, mejor_v = float(u), val
    p["umbral_dispersion"] = mejor_u
    print(f"umbral de acuerdo elegido en validacion: {mejor_u:.3f} "
          f"({(disp <= mejor_u).mean():.1%} de las tareas) · PRED25 val {mejor_v:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(p, V4, compress=3)
    print(f"escrito {V4} ({V4.stat().st_size / 1e6:.0f} MB) · {time.time() - t0:.0f} s")

    # verificacion sobre los bloques congelados, sin volver a codificar nada
    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    salidas = {"horas (apilado)": np.exp(lp),
               "valor_probable (analogia encaminada)": np.where(disp <= mejor_u, moda, np.exp(lp))}
    filas = []
    for b, k in bl.items():
        for n, q in salidas.items():
            filas.append(dict(bloque=b, salida=n, n=int(k.sum()), **metricas(y[k], q[k])))
    t = pd.DataFrame(filas)
    cols = ["salida", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "verificacion.csv", index=False)
    json.dump({"k": K_VEC, "umbral_dispersion": mejor_u, "atomos": len(rej)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/verificacion.csv")


if __name__ == "__main__":
    construir()
