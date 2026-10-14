"""E64 - El estimador v6: el que se integra en Agix.

Junta en un objeto todo lo que sobrevivio a su medicion, y separa con claridad las
dos preguntas que el producto hace, porque NO tienen la misma respuesta:

  CUANTO TARDA ESTA TAREA   quiere la mediana condicional. Es el numero que se le
                            ensena a una persona al lado de una tarea
  CUANTO TARDA EL PROYECTO  quiere la MEDIA, porque se va a sumar. Sumar medianas
                            deja el total en el 44 % de lo que toca

Por eso lleva dos modelos y no uno, entrenados con perdidas distintas sobre los
mismos rasgos. Confundirlos fue el fallo que aparecio tres veces esta noche.

Lo que hace:

    estimar_tareas(textos)                     horas por tarea, con su banda
    estimar_proyecto(textos, cerradas=None,    total del proyecto, con intervalo
                     estimaciones=None)        al 80 % y anclaje si hay historial

Detalles que importan y que costaron encontrar:

  - el anclaje con tareas cerradas usa el COCIENTE DE SUMAS, no la mediana de los
    cocientes, y contrae con James-Stein (lambda 12). Con la mediana, cuantas mas
    tareas cerradas peor: el sesgo llegaba a -45 %
  - las tareas ya cerradas se SABEN. El total es lo conocido mas la estimacion
    corregida de lo que queda
  - si el usuario estima, se corrige su numero con un modelo de Poisson, no con el
    corrector cuadratico: a nivel de proyecto el cuadratico EMPEORA la estimacion
    cruda del usuario
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
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
V5 = Path("models/estimador_agix_v5.joblib")
V6 = Path("models/estimador_agix_v6.joblib")
OUT = Path("results/estimador_v6")
S = "=" * 116
N_BOLSA, K_VEC, LAMBDA = 5, 25, 12
# La contraccion del anclaje depende de cuanto desplazamiento de proyecto quede por
# corregir, y eso cambia si el usuario ha estimado. Medido en E85/E86: en frio el
# residuo tiene un 11,1 % de su varianza ENTRE proyectos, pero tras aplicar la
# estimacion del usuario solo queda un 1,9 %. Con tan poco que corregir, el anclaje
# con lambda = 12 mete mas ruido que senal: anclar 3 tareas sobre la estimacion
# BAJABA el +-25 % del 73,2 % al 70,5 %. Con lambda = 96 ya no empeora en ningun k
# y el mejor caso sube del 79,9 % al 87,4 %.
LAMBDA_CON_ESTIMACION = 96
# cortes de tamano para calibrar la banda por tramo, y cuantos proyectos hacen
# falta en un tramo para fiarse de el en vez de caer a la banda unica
CORTES_TRAMO = [20, 60, 200]
N_REJILLA = 30
# contraccion del ancla por persona; barrida en E100 sobre 423 desarrolladores
LAMBDA_PERSONA = 8
MIN_CALIBRACION = 12


def qconf(x, p, arriba):
    """Cuantil CONFORME: el estadistico de orden ceil((m+1)p), no np.quantile.

    Es lo que da la garantia de cobertura con m puntos de calibracion. Con el
    cuantil muestral la banda cubre de MENOS, y el sesgo crece cuanto mas
    pequeno es el grupo: el 80 % nominal salia 78,1 % con los 114 proyectos y
    74,6 % calibrando por tramo. Con esta correccion salen 80,7 % y 82,5 %.
    """
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def poi(semilla=0, **kw):
    p = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
             loss="poisson", random_state=semilla)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


class EstimadorV6:
    def __init__(self, paquete=V6, dispositivo=None):
        self.p = joblib.load(paquete)
        self._st, self._aj, self._ac, self._dev = None, None, None, dispositivo

    def _cod(self, ruta, attr, maxlen=48):
        if getattr(self, attr) is None:
            import torch
            from sentence_transformers import SentenceTransformer, models
            dv = self._dev or ("cuda" if torch.cuda.is_available() else "cpu")
            if ruta.startswith("intfloat"):
                setattr(self, attr, SentenceTransformer(ruta, device=dv))
            else:
                w = models.Transformer(ruta, max_seq_length=maxlen)
                setattr(self, attr, SentenceTransformer(
                    modules=[w, models.Pooling(w.get_word_embedding_dimension(), "mean")],
                    device=dv))
        return getattr(self, attr)

    def _vistas(self, textos):
        V = self._cod(self.p["codificador"], "_st").encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        Ms = sparse.hstack([self.p["tfc"].transform(textos),
                            self.p["tfw"].transform(textos)]).tocsr()
        SV = np.hstack([sv.transform(tf.transform(textos)).astype(np.float32)
                        for tf, sv in zip((self.p["tfc"], self.p["tfw"]),
                                          self.p["svds"])])
        return V, np.hstack([V, SV]), SV, Ms

    def _analogia(self, V):
        Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        s = Vn @ self.p["vec_V"].astype(np.float32).T
        j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
        v = self.p["vec_y"][j]
        med = np.median(v, axis=1)
        rej = self.p["rejilla"]
        moda = rej[np.abs(np.log(rej)[None, :] -
                          np.log(np.clip(med, 1e-6, None))[:, None]).argmin(1)]
        return moda, np.std(np.log(np.clip(v, 1e-6, None)), axis=1)

    def _offsets_persona(self, personas, historial):
        """Un desplazamiento por persona, a partir de SUS tareas ya cerradas.

        Mismo mecanismo que el anclaje de proyecto -cociente de sumas contraido
        con k/(k+lambda)- pero por quien hace la tarea. E100 descompuso el
        residuo en tres niveles y encontro que entre DESARROLLADORES del mismo
        proyecto hay un 9,8 % de la varianza, casi tanto como entre proyectos
        (12,4 %), y ese trozo estaba dentro del 89 % que se daba por ruido.

        Solo sirve en EQUIPO: en un proyecto de una sola persona este
        desplazamiento y el del proyecto son el mismo numero.
        """
        quien = sorted({p for p in personas if p is not None and p in historial
                        and len(historial[p]) > 0})
        if not quien:
            return {}
        # se codifica todo el historial de una vez, no persona a persona
        plano, corte = [], {}
        for q in quien:
            i0 = len(plano)
            plano.extend(t for t, _ in historial[q])
            corte[q] = (i0, len(plano))
        h = self._apilado_crudo(plano)
        off = {}
        for q in quien:
            i0, i1 = corte[q]
            real = np.array([v for _, v in historial[q]], float)
            k = len(real)
            off[q] = float(np.log(real.sum() / max(h[i0:i1].sum(), 1e-9))
                           * k / (k + LAMBDA_PERSONA))
        return off

    def _apilado_crudo(self, textos):
        """Las horas del apilado, sin factor de estilo ni rejilla."""
        V, Xd, SV, Ms = self._vistas(textos)
        aj = self._cod(self.p["codificador_ajustado"], "_aj").encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        ac = self._cod(self.p["codificador_ajustado_C"], "_ac").encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        Z = np.column_stack([np.mean([m.predict(Xd) for m in self.p["bolsa"]], 0),
                             self.p["ridge"].predict(Ms), self.p["svr"].predict(Ms),
                             self.p["hgb_ajustado"].predict(aj),
                             self.p["hgb_ajustado_C"].predict(ac)])
        return np.exp(self.p["apilado"].predict(Z))

    def estimar_tareas(self, textos, estilo="persona", personas=None,
                       historial=None, estimaciones=None):
        """La mediana condicional: el numero que se le ensena a una persona.

        estilo: "persona" para texto escrito por alguien -lo normal- y
        "generador" para texto que viene de la tuberia de generacion.

        personas: quien hara cada tarea, en el mismo orden que textos, con None
        donde no se sepa. historial: {persona: [(texto, horas_reales), ...]} con
        sus tareas ya cerradas. Si se pasan los dos, cada tarea se corrige con
        el desplazamiento de quien la hara.

        Sobre la vista por tarea esto mejora el orden y el error absoluto
        (spearman 0,317 -> 0,358, SA 0,026 -> 0,036, MAE 5,06 -> 5,00 h). NO se
        usa para el total del proyecto: ahi los desplazamientos de las personas
        se cancelan al sumar y empeoran el pronostico (E101).

        estimaciones: lo que estima el usuario, en el orden de textos, con None
        donde no estime. Donde la hay, las horas salen del corrector por tarea y
        la banda es la suya: x10,4 de anchura en lugar de x32, con la misma
        cobertura del 80 % medida sin las filas de real copiado (E108). Con solo
        el texto la banda NO se puede estrechar: ni por acuerdo de vecinos, ni
        con cuantiles aprendidos, ni por valor predicho.

        El factor de estilo se midio en E74 sobre DOS corpus: el texto original
        de Jira sale con sesgo x1,002, es decir ninguno, y el del generador con
        x1,378. El factor corrige al generador. Aplicarlo tambien al texto de
        persona -que es lo que se hacia- encogia la prediccion un 17 % sin
        motivo: el sesgo mediano del artefacto sobre texto real era x0,820 en
        lugar de 1,0 (E97). Quitarlo deja el sesgo en x0,987, y con el ajuste a
        la rejilla en x1,000 exacto (E98).
        """
        textos = list(textos)
        V, Xd, SV, Ms = self._vistas(textos)
        aj = self._cod(self.p["codificador_ajustado"], "_aj").encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        ac = self._cod(self.p["codificador_ajustado_C"], "_ac").encode(
            ["query: " + t for t in textos], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        Z = np.column_stack([np.mean([m.predict(Xd) for m in self.p["bolsa"]], 0),
                             self.p["ridge"].predict(Ms), self.p["svr"].predict(Ms),
                             self.p["hgb_ajustado"].predict(aj),
                             self.p["hgb_ajustado_C"].predict(ac)])
        fe = self.p["factor_estilo"] if estilo == "generador" else 1.0
        h = np.exp(self.p["apilado"].predict(Z) + np.log(fe))
        # el desplazamiento de quien hara la tarea, si se sabe
        ajuste = np.ones(len(textos))
        if personas is not None and historial:
            off = self._offsets_persona(personas, historial)
            ajuste = np.array([np.exp(off.get(q, 0.0)) if q is not None else 1.0
                               for q in personas])
            h = h * ajuste
        ana, disp = self._analogia(V)
        k = self.p["factor_banda80"]
        # El valor probable se ajusta SIEMPRE a la rejilla, tambien cuando los
        # vecinos no concuerdan y se cae a la prediccion del apilado. Las horas
        # que la gente registra son un peine -treinta valores cubren la mitad
        # del corpus- y PRED(25) premia caer encima de un diente: ajustar sube
        # el +-25 % de 15,5 % a 17,1 % y el factor 2 de 0,403 a 0,424 sin mover
        # el MAE, el SA ni el orden entre tareas (E95, E96).
        # el desplazamiento de la persona se aplica tambien a la analogia: es una
        # propiedad de quien hace la tarea, no del canal que la estimo
        crudo = np.where(disp <= self.p["umbral_dispersion"], ana * ajuste, h)
        lo_b, hi_b = h / k, h * k
        con_est = np.zeros(len(textos), bool)
        if estimaciones is not None and "corrector_tarea_v6" in self.p:
            e_ = np.array([np.nan if v is None else float(v) for v in estimaciones])
            con_est = ~np.isnan(e_) & (np.nan_to_num(e_) > 0)
            if con_est.any():
                Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
                Xc = np.hstack([Vn, np.log(np.where(con_est, e_, 1))[:, None]])
                hc = np.exp(self.p["corrector_tarea_v6"].predict(Xc)) * ajuste
                bl, bh = self.p["banda_con_estimacion"]
                h = np.where(con_est, hc, h)
                crudo = np.where(con_est, hc, crudo)
                lo_b = np.where(con_est, hc * np.exp(bl), lo_b)
                hi_b = np.where(con_est, hc * np.exp(bh), hi_b)
        rej = self.p["rejilla"]
        probable = rej[np.abs(np.log(rej)[None, :] -
                              np.log(np.clip(crudo, 1e-6, None))[:, None]).argmin(1)]
        return pd.DataFrame({"texto": textos, "horas": h,
                             "valor_probable": probable,
                             "banda_baja": lo_b, "banda_alta": hi_b,
                             "ajuste_persona": ajuste, "con_estimacion": con_est})

    def _media_tarea(self, textos):
        """La media condicional: lo que hay que sumar para un total."""
        V, Xd, SV, Ms = self._vistas(textos)
        ana, _ = self._analogia(V)
        C = np.column_stack([np.mean([m.predict(Xd) for m in self.p["tot_bolsa"]], 0),
                             self.p["tot_sv"].predict(SV),
                             self.p["tot_ana"].predict(ana[:, None]),
                             self.p["tot_lin"].predict(Ms)])
        return np.clip(self.p["tot_apilado"].predict(C), 1e-6, None)

    def estimar_proyecto(self, textos, cerradas=None, estimaciones=None, nivel=.8):
        """cerradas: horas REALES ya registradas, None donde no las haya.

        estimaciones: lo que el usuario estimo, None donde no haya estimado."""
        textos = list(textos)
        m = self._media_tarea(textos)
        cer = (np.array([np.nan if c is None else float(c) for c in cerradas])
               if cerradas is not None else np.full(len(textos), np.nan))
        hay = ~np.isnan(cer)
        uso_estimacion = False
        if estimaciones is not None:
            est = np.array([np.nan if e is None else float(e) for e in estimaciones])
            he = ~np.isnan(est)
            if he.any():
                V, _, _, _ = self._vistas(textos)
                Xc = np.hstack([V, np.log(np.where(he, est, 1))[:, None],
                                np.log(m)[:, None]])
                m = np.where(he, self.p["corrector_total"].predict(Xc), m)
                uso_estimacion = True
        k = int(hay.sum())
        lam = LAMBDA_CON_ESTIMACION if uso_estimacion else LAMBDA
        if k:
            off = np.log(cer[hay].sum() / m[hay].sum()) * k / (k + lam)
            cerrado = float(cer[hay].sum())
            abierto = float((m[~hay] * np.exp(off)).sum())
        else:
            cerrado, abierto = 0.0, float(m.sum())
        total = cerrado + abierto
        # La banda va SOLO sobre lo que queda abierto: las horas cerradas se
        # saben y no tienen incertidumbre. Y se calibra al numero de tareas
        # cerradas y al tramo de tamano del proyecto. Antes multiplicaba el total
        # entero por el residuo en frio y no se estrechaba nunca: x10,61 con 0 o
        # con 20 tareas cerradas, cubriendo el 98 % al final. Ahora baja a x4,96
        # con 3, x3,75 con 5, x2,37 con 10 y x2,05 con 20, manteniendo el 80 %
        # (E107). Por tramo se conserva el reparto honesto de E89.
        tramo = int(np.digitize(len(textos), self.p["cortes_tramo"]))
        tabla = self.p.get("residuos_abierto")
        if tabla:
            # el k calibrado mas cercano por DEBAJO: con menos historial la banda
            # es mas ancha, asi que redondear hacia abajo es lo prudente
            kb = max(x for x in self.p["ks_banda"] if x <= k)
            g = tabla[kb]
            if len(g.get(tramo, ())) >= MIN_CALIBRACION:
                q = g[tramo]
            else:
                q, tramo = g[-1], None
        else:
            kb = 0
            q = self.p["residuos_proyecto"]
            g = self.p.get("residuos_por_tramo") or {}
            if len(g.get(tramo, ())) >= MIN_CALIBRACION:
                q = g[tramo]
            else:
                tramo = None
        a = (1 - nivel) / 2
        return dict(total_horas=total, horas_cerradas=cerrado,
                    horas_abiertas=abierto,
                    banda_baja=float(cerrado + abierto * np.exp(qconf(q, a, False))),
                    banda_alta=float(cerrado + abierto *
                                     np.exp(qconf(q, 1 - a, True))),
                    nivel=nivel, tareas=len(textos), cerradas=k,
                    con_estimacion=uso_estimacion, lambda_usada=lam,
                    tramo_banda=tramo, k_banda=kb, n_calibracion=len(q))


def construir():
    t0 = time.time()
    p = dict(joblib.load(V5))
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    tr = (sw.split_t3 == "train").to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((p["tfc"], p["tfw"]), p["svds"])])
    X = np.hstack([E5, SV])
    Xs = sparse.hstack([p["tfc"].transform(tit), p["tfw"].transform(tit)]).tocsr()

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr)
    bolsa = []
    for s in range(N_BOLSA):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(poi(s, max_features=.8).fit(X[sub], y[sub]))
    m_sv = poi().fit(SV[tr], y[tr])
    A, yv = Vn[tr], y[tr]

    def vecino(sel):
        out = np.zeros(len(sel))
        for i in range(0, len(sel), 2048):
            s = Vn[sel[i:i + 2048]] @ A.T
            j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
            out[i:i + 2048] = np.median(yv[j], axis=1)
        return out
    an = vecino(np.arange(len(sw)))
    m_an = poi().fit(an[tr][:, None], y[tr])
    m_li = PoissonRegressor(alpha=1e-3, max_iter=400).fit(Xs[tr], y[tr])
    C = np.column_stack([np.mean([m.predict(X) for m in bolsa], 0),
                         m_sv.predict(SV), m_an.predict(an[:, None]),
                         m_li.predict(Xs)])
    va = (sw.split_t3 == "val").to_numpy()
    ap = poi(max_iter=120, max_depth=3).fit(C[va], y[va])
    print(f"modelo del total ajustado · {time.time() - t0:.0f} s", flush=True)

    est = np.array(sw.HoursEstimate.where(sw.HoursEstimate > 0), dtype=float)
    hay = ~np.isnan(est)
    Xc = np.hstack([E5, np.log(np.where(hay, est, 1))[:, None],
                    np.log(np.clip(ap.predict(C), 1e-6, None))[:, None]])
    k = hay & tr
    corr = poi().fit(Xc[k], y[k])

    p.update({"tot_bolsa": bolsa, "tot_sv": m_sv, "tot_ana": m_an, "tot_lin": m_li,
              "tot_apilado": ap, "corrector_total": corr,
              "vec_V": Vn[tr].astype(np.float16), "vec_y": y[tr].astype(np.float32)})

    # La rejilla de valores frecuentes y el umbral de acuerdo entre vecinos venian
    # del v4; el v5 se construye sobre el v3 y no los hereda. Sin ellos el
    # artefacto se construye igual y revienta al estimar, que es justo lo que
    # destapo la prueba de extremo a extremo.
    # El tamano de la rejilla se barrio fuera de pliegue en E96: 30 valores por
    # frecuencia es el optimo. El umbral de 60 apariciones daba 41 valores y
    # rendia medio punto menos de +-25 % (16,59 % contra 17,11 %).
    cuentas = pd.Series(y[tr]).value_counts()
    rej = np.sort(cuentas.index.to_numpy()[:N_REJILLA])
    p["rejilla"] = rej
    A2, yv2 = Vn[tr], y[tr]
    ana = np.zeros(len(sw))
    disp = np.zeros(len(sw))
    for i in range(0, len(sw), 2048):
        sm = Vn[i:i + 2048] @ A2.T
        j = np.argpartition(-sm, K_VEC, axis=1)[:, :K_VEC]
        v = yv2[j]
        med = np.median(v, axis=1)
        ana[i:i + 2048] = rej[np.abs(np.log(rej)[None, :] -
                                     np.log(np.clip(med, 1e-6, None))[:, None]).argmin(1)]
        disp[i:i + 2048] = np.std(np.log(np.clip(v, 1e-6, None)), axis=1)
    lp_v5 = p["apilado"].predict(np.column_stack([
        np.mean([m.predict(np.hstack([E5, SV])) for m in p["bolsa"]], 0),
        p["ridge"].predict(Xs), p["svr"].predict(Xs),
        p["hgb_ajustado"].predict(np.load(CACHE / "e5-ajustado.npy")),
        p["hgb_ajustado_C"].predict(np.load(CACHE / "e5-ajustado-C.npy"))]))
    va2 = (sw.split_t3 == "val").to_numpy()
    mejor_u, mejor_v = float(np.median(disp)), -1.
    for u in np.quantile(disp[va2], np.arange(.05, 1., .05)):
        q = np.where(disp <= u, ana, np.exp(lp_v5))
        v_ = float(np.mean(np.abs(q[va2] - y[va2]) / y[va2] <= .25))
        if v_ > mejor_v:
            mejor_u, mejor_v = float(u), v_
    p["umbral_dispersion"] = mejor_u
    print(f"rejilla de {len(rej)} valores · umbral de acuerdo {mejor_u:.3f} "
          f"({(disp <= mejor_u).mean():.1%} de las tareas) · PRED25 val {mejor_v:.4f}",
          flush=True)
    tot = pd.read_csv("results/curva_final/totales.csv")
    res_p = np.log(tot.real.to_numpy() / tot.frio.to_numpy())
    p["residuos_proyecto"] = res_p
    gr = np.digitize(tot.n.to_numpy(), CORTES_TRAMO)
    p["residuos_por_tramo"] = {int(t): res_p[gr == t] for t in np.unique(gr)}
    p["cortes_tramo"] = CORTES_TRAMO
    print("  banda por tramo: " + " · ".join(
        f"{t}:n={len(v)}" for t, v in sorted(p["residuos_por_tramo"].items())),
        flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(p, V6, compress=3)
    print(f"escrito {V6} ({V6.stat().st_size / 1e6:.0f} MB) · {time.time() - t0:.0f} s")
    json.dump({"n_bolsa": N_BOLSA, "k_vec": K_VEC, "lambda": LAMBDA},
              open(OUT / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    construir()
