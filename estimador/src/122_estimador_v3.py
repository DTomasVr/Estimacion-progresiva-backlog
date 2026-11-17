"""El estimador v3: el modelo final de la noche, con toda la capa de servicio.

Junta en un solo objeto lo mejor de cada cosa que sobrevivio a su medicion:

  MODELO        bolsa de 7 HGB sobre e5-large + tf-idf reducido, apilada con un
                ridge y un SVR lineal sobre la matriz dispersa entera
                (`models/modelo_final.joblib`, construido por el script 114)
  ANCLAS        desplazamiento del proyecto con contraccion James-Stein
                (lambda = 12). Sin contraccion, anclar con pocas tareas EMPEORA
  CORRECTOR     si el usuario estima, se corrige su numero. Es la palanca
                grande: R² 0,063 -> 0,340 en proyectos nunca vistos
  FAMILIARIDAD  nivel de parecido con el historial del proyecto, que es la mejor
                señal de confianza medida y la unica que mejora sola con el uso
  BANDA         conforme al 80 %, calibrada en validacion
  SMEARING      para el total del proyecto, que no es la suma de las medianas
  ESTILO        factor para el texto que escribe el generador de tareas

    from importlib.util import spec_from_file_location, module_from_spec
    s = spec_from_file_location("e", "src/122_estimador_v3.py")
    m = module_from_spec(s); s.loader.exec_module(m)
    est = m.EstimadorV3()
    r = est.estimar(["Store payment data in the member profile"],
                    proyecto=["p1"], estimacion_usuario=[None], anclas=[None])
    est.total_proyecto(r)
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
FINAL = Path("models/modelo_final.joblib")
PAQUETE = Path("models/estimador_pbe_v3.joblib")
OUT = Path("results/estimador_v3")
SEP = "=" * 126
NIVELES = [(.97, "ya hiciste algo casi igual"), (.93, "algo parecido"),
           (.85, "de la misma familia"), (-2, "esto es nuevo")]


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


class EstimadorV3:
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

    @property
    def codificador_ajustado(self):
        if getattr(self, "_aj", None) is None:
            import torch
            from sentence_transformers import SentenceTransformer, models
            d = self._dev or ("cuda" if torch.cuda.is_available() else "cpu")
            w = models.Transformer(self.p["codificador_ajustado"], max_seq_length=48)
            self._aj = SentenceTransformer(
                modules=[w, models.Pooling(w.get_word_embedding_dimension(), "mean")],
                device=d)
        return self._aj

    def _ajustado(self, textos):
        return self.codificador_ajustado.encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)

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

    def estimar(self, textos, proyecto=None, estimacion_usuario=None, anclas=None,
                historial=None, estilo_generador=True):
        """Una fila por tarea.

        historial: lista de textos ya cerrados del proyecto, para el nivel de
                   familiaridad. Si falta, se usa el corpus de entrenamiento.
        """
        textos = list(textos)
        n = len(textos)
        V, Xd, Ms = self._vistas(textos)
        Z = np.column_stack([np.mean([m.predict(Xd) for m in self.p["bolsa"]], 0),
                             self.p["ridge"].predict(Ms), self.p["svr"].predict(Ms),
                             self.p["hgb_ajustado"].predict(self._ajustado(textos))])
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
                             "banda_baja": horas / k80, "banda_alta": horas * k80,
                             "familiaridad": sim, "nivel": nivel})

    def total_proyecto(self, r):
        s = self.p["smearing"]
        total = float(r.horas.sum() * s)
        return {"tareas": len(r), "suma_de_medianas_h": float(r.horas.sum()),
                "total_esperado_h": total, "total_esperado_dias": total / 8,
                "total_ajustado_h": total / self.p["suma_residual"]}


def construir():
    f = joblib.load(FINAL)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    tr_m = (sw.split_t3 == "train").to_numpy()
    va_m = (sw.split_t3 == "val").to_numpy()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    E5 = np.load(CACHE / "e5-large.npy")
    Ms = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Xd = np.hstack([E5] + [sv.transform(tf.transform(tit)).astype(np.float32)
                           for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    AJ = np.load(CACHE / "e5-ajustado.npy")
    Z = np.column_stack([np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Ms), f["svr"].predict(Ms),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)

    res = np.abs(ly[va_m] - lp[va_m])
    banda80 = float(np.exp(np.quantile(res, .8)))
    S = float(np.mean(np.exp(ly[tr_m] - lp[tr_m])))
    suma_res = float((np.exp(lp[va_m]) * S).sum() / y[va_m].sum())
    corr = json.load(open("results/correcciones.json", encoding="utf-8"))
    Xn = E5 / np.linalg.norm(E5, axis=1, keepdims=True)

    # corrector C1 (estimacion + frio + TEXTO): E24 lo midio mejor que el C2 que
    # traia el modelo final, sobre todo donde lo real difiere de lo estimado
    # (R2 0,4745 frente a 0,4607)
    sw2 = sw.copy()
    sw2["est"] = sw2.HoursEstimate.where(sw2.HoursEstimate > 0)
    pt = sw2[(sw2.source == "TAWOS") & (sw2.story_point > 0)]
    ptr = pt[pt.split_t3 == "train"]
    pend, corte = np.polyfit(np.log(ptr.story_point.to_numpy(float)),
                             np.log(ptr.effort_hours.to_numpy(float)), 1)
    sw2.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    est = sw2.est.to_numpy(float)
    hay = ~np.isnan(est)
    from sklearn.model_selection import KFold
    oof = lp.copy()
    it = np.flatnonzero(tr_m)
    for a_, b_ in KFold(5, shuffle=True, random_state=0).split(it):
        oof[it[b_]] = HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=.06, l2_regularization=1.,
            random_state=0).fit(Xd[it[a_]], ly[it[a_]]).predict(Xd[it[b_]])
    kk = tr_m & hay
    Zc = np.hstack([E5, np.log(np.where(hay, est, 1))[:, None], oof[:, None]])
    corrector = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=.06, l2_regularization=1.,
        random_state=0).fit(Zc[kk], (ly - np.log(np.where(hay, est, 1)))[kk])
    print("  corrector C1 (con texto) ajustado")

    paq = dict(f)
    paq["corrector"] = corrector
    paq["corrector_variante"] = "C1 (vector + log estimacion + frio)"
    paq.update({"factor_banda80": banda80, "smearing": S, "suma_residual": suma_res,
                "factor_estilo": corr["estilo_medio"],
                "centros": Xn[tr_m].astype(np.float32),
                "idioma_esperado": "ingles", "vista": "titulo",
                "fecha": time.strftime("%Y-%m-%d")})
    PAQUETE.parent.mkdir(exist_ok=True)
    joblib.dump(paq, PAQUETE, compress=3)
    print(f"escrito {PAQUETE} ({PAQUETE.stat().st_size / 1e6:.0f} MB)")
    print(f"  banda 80 % ×{banda80:.2f} · smearing {S:.3f} · suma residual {suma_res:.3f}")
    return sw, lp, y, ly


def verificar(sw, lp, y, ly):
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = [dict(bloque=b, n=int(k.sum()), **metricas(y[k], np.exp(lp[k])))
             for b, k in bloques.items()]
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nVERIFICACIÓN DEL v3\n{SEP}")
    print(t.round(4).to_string(index=False))
    Path(OUT).mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "verificacion.csv", index=False)
    print(f"\nescrito {OUT}/verificacion.csv")


if __name__ == "__main__":
    sw, lp, y, ly = construir()
    verificar(sw, lp, y, ly)
