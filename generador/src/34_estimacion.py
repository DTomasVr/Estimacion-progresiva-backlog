"""Une el generador de backlog con el estimador de esfuerzo (estimador_agix_v7).

El generador produce historias con sus tareas; este modulo les pone horas. Son dos
modelos distintos que viven en dos repositorios distintos, asi que aqui esta todo lo
que hace falta para que hablen, y nada mas.

DOS NUMEROS, NO UNO. El estimador lleva dos cabezas a proposito, y aqui se usan las
dos para lo que sirve cada una:

  horas de UNA tarea     la MEDIANA condicional (cabeza A). Es el numero que se le
                         ensena a una persona al lado de una tarea.
  cualquier SUMA         la MEDIA condicional (cabeza B). Sumar medianas deja el
                         total en el 44 % de lo que toca.

Por eso el subtotal de una historia NO es la suma de las horas que se muestran en
sus tareas, y eso no es un fallo: son dos preguntas distintas. La interfaz lo dice.

EL TEXTO VA EN INGLES. El estimador se entreno con titulos de Jira en ingles y su
contrato lo exige. El generador trabaja en ingles internamente y solo traduce al
final, asi que aqui se usa h["tareas"] (el ingles) y nunca h["tareas_es"]. Pasarle
el espanol seria salirse del contrato sin que nada fallase: solo acertaria menos.

    from importlib import util
    e = Estimacion()                 # un solo codificador ONNX de 397 MB, en CPU
    r = e.estimar(generador(desc))   # anade horas a r
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

# Donde esta el estimador. Se busca en este orden, para que la misma copia sirva
# tanto en la maquina de desarrollo como en una carpeta portatil:
#   1 la variable AGIX_DATASET, si esta puesta
#   2 una carpeta "estimador" al lado de este modulo (el reparto para la laptop)
#   3 el repositorio Agix-Dataset en la maquina de desarrollo
def _raiz_estimador():
    if os.environ.get("AGIX_DATASET"):
        return Path(os.environ["AGIX_DATASET"])
    aqui = Path(__file__).resolve().parent.parent / "estimador"
    if any((aqui / "models" / f"estimador_agix_{v}.joblib").exists() for v in ("v7", "v6")):
        return aqui
    return Path(r"C:\Users\User\Documents\Programacao\Agix-Dataset")


RAIZ = _raiz_estimador()

# EL CODIFICADOR VA DENTRO DEL ESTIMADOR. Desde el v7 el estimador usa UN solo
# codificador -el e5-large base, en ONNX int8 con el vocabulario podado- guardado en
# estimador/models/e5_v7 con su tokenizador. No hace falta el cache de Hugging Face
# ni red. Con el v6 si: el base venia del cache y habia dos codificadores ajustados
# mas (6,5 GB en total); se mantiene la compatibilidad por si se vuelve a el.

# EL FACTOR DE ESTILO DEL TOTAL. El texto del generador es mas escueto que un titulo
# de Jira y eso infla las predicciones. El artefacto trae un factor de 0,8313, pero
# esta medido sobre la cabeza de la MEDIANA; la cabeza de la MEDIA, que es de donde
# sale el total, no lo aplica y tiene otro.
#
# Medido en E74 (Agix-Dataset/src/182_estilo_para_el_total.py) sobre las 800 tareas
# reales reescritas al estilo del generador, todas fuera del entrenamiento, con el
# cociente de sumas que es lo que corresponde a un total:
#
#   titulo original de Jira   la suma predicha es 1,0018 de la real  (calibrado)
#   reescrito al estilo       la suma predicha es 1,3775 de la real  (+37,8 %)
#   factor que lo corrige     0,7273
#
# Sin esto el total del proyecto sale un 37 % alto. Se lee del fichero de resultados
# si esta, y si no se usa el valor medido.
FACTOR_ESTILO_TOTAL = 0.7273


def _factor_estilo_total(raiz):
    """El factor medido, leido de su fichero de resultados si esta disponible."""
    f = Path(raiz) / "results" / "estilo_total" / "meta.json"
    try:
        import json
        return float(json.load(open(f))["factor_total"]), str(f)
    except Exception:                                # noqa: BLE001
        return FACTOR_ESTILO_TOTAL, "valor medido por defecto"


def hay_codificador_base(nombre="intfloat/multilingual-e5-large", raiz=None):
    """¿Esta el codificador base? v7: en el propio estimador. v6: en el cache de HF."""
    if ((Path(raiz) if raiz else RAIZ) / "models" / "e5_v7" / "model.int8.onnx").exists():
        return True
    try:
        from huggingface_hub import constants as c
        carpeta = "models--" + nombre.replace("/", "--")
        return (Path(c.HF_HUB_CACHE) / carpeta).exists()
    except Exception:                                # noqa: BLE001
        return False


# Factores de escala del bloque Sprint 0 (base tecnica: proyecto, datos, acceso, CI/CD...,
# ver plan_sprint0 en 19_pipeline.py). Medidos el 12/11/2026 contra tareas REALES de ese tipo:
# 522 tareas de base tecnica del corpus que el estimador no vio (split test y val), las 15 mas
# parecidas a cada plantilla con el SBERT del proyecto.
#   - Sobre esas tareas reales el estimador ACIERTA (media real 5,85 h, modelo 6,02 h).
#   - Pero las plantillas son titulos largos y genericos, y el modelo les da mas del doble
#     que a sus vecinas reales: media 11,45 h frente a 5,83 h.
# El factor se aplica en bloque y no por plantilla, porque plantilla a plantilla hay ruido
# (la vecindad media es 0,44). Sin el, el Sprint 0 sumaria un 61 % al total del proyecto en
# lugar de un 31 %. IC95 del factor de la media: [0,40, 0,64].
FACTOR_SPRINT0_MEDIA = 0.509      # cabeza B: la que suma el total
FACTOR_SPRINT0_MEDIANA = 0.450    # cabeza A: la que se ensena por tarea


class Estimacion:
    """Carga el estimador (v7, o v6 si no esta) y le pone horas a un backlog generado.

    v7: un solo codificador, asi que por_tarea ya no cuesta memoria extra; solo
    decide si se calculan y devuelven las horas de cada tarea. v6: por_tarea=True
    cargaba ademas los dos codificadores ajustados (2,2 GB cada uno).
    """

    def __init__(self, raiz=RAIZ, por_tarea=True, dispositivo=None, nivel=.8,
                 estilo=True):
        self.raiz, self.por_tarea, self.nivel = Path(raiz), por_tarea, nivel
        self.disponible, self.motivo, self.carga_s = False, "", 0.
        self.est = None
        self.factor_estilo, self.fuente_estilo = _factor_estilo_total(raiz)
        if not estilo:                               # para comparar, no para servir
            self.factor_estilo, self.fuente_estilo = 1.0, "desactivado"
        t0 = time.time()
        paquete = self.raiz / "models" / "estimador_agix_v7.joblib"
        if not paquete.exists():
            paquete = self.raiz / "models" / "estimador_agix_v6.joblib"
        guion = self.raiz / "src" / "171_estimador_v6.py"
        for f in (paquete, guion):
            if not f.exists():
                self.motivo = f"falta {f}"
                return
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("est_v6", guion)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["est_v6"] = mod
            spec.loader.exec_module(mod)
            self.est = mod.EstimadorV6(paquete=paquete, dispositivo=dispositivo)
            # v6: el paquete guarda las rutas de los codificadores ajustados RELATIVAS
            # a la raiz del otro repositorio (el v7 ya no los tiene, y sus rutas las
            # resuelve la propia clase). Aqui el directorio de trabajo es otro, asi
            # que hay que resolverlas o SentenceTransformer intentaria bajarlas de
            # Hugging Face con ese nombre y fallaria.
            for k in ("codificador_ajustado", "codificador_ajustado_C"):
                v = self.est.p.get(k)
                if v and not str(v).startswith("intfloat"):
                    self.est.p[k] = str((self.raiz / v).resolve())
            self.disponible = True
        except Exception as e:                       # noqa: BLE001
            self.motivo = f"{type(e).__name__}: {e}"
        self.carga_s = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ interno
    @staticmethod
    def _qconf(x, p, arriba):
        """Cuantil CONFORME: el estadistico de orden ceil((m+1)p).

        No np.quantile: con el cuantil muestral la banda cubre de menos (78,1 %
        en vez de 80 % con los 114 proyectos, y 74,6 % calibrando por tramo).
        """
        s = np.sort(np.asarray(x, float))
        m = len(s)
        j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
        return float(s[int(np.clip(j, 0, m - 1))])

    def _banda(self, cerrado, abierto, n_tareas, k):
        """El intervalo del total, con la misma regla que estimar_proyecto.

        Se calcula aqui para no codificar el texto dos veces: estimar_proyecto
        volveria a pasar todas las tareas por el codificador solo para esto.

        La incertidumbre va SOLO sobre lo abierto -lo cerrado se sabe- y se
        calibra al numero de tareas cerradas y al tramo de tamano. Antes la
        banda multiplicaba el total entero por el residuo en frio y no se
        estrechaba nunca (x10,61 con 0 o con 20 cerradas, E107).
        """
        t = int(np.digitize(n_tareas, self.est.p["cortes_tramo"]))
        a = (1 - self.nivel) / 2
        tabla = self.est.p.get("residuos_abierto")
        if tabla:
            kb = max(x for x in self.est.p["ks_banda"] if x <= k)
            g = tabla[kb]
            qt = g.get(t, ())
            if len(qt) >= 12:
                # el cuantil del tramo contraido hacia el global: calibrar cada
                # tramo aparte cuesta anchura cuando hay pocas cerradas (E114)
                M = {0: 1500, 3: 600, 5: 200, 10: 0, 20: 0}.get(kb, 0)
                w = len(qt) / (len(qt) + M) if M else 1.0
                ql = w * self._qconf(qt, a, False) + (1 - w) * self._qconf(g[-1], a, False)
                qh = w * self._qconf(qt, 1 - a, True) + (1 - w) * self._qconf(g[-1], 1 - a, True)
                return (float(cerrado + abierto * np.exp(ql)),
                        float(cerrado + abierto * np.exp(qh)))
            q = g[-1]
        else:
            q = self.est.p["residuos_proyecto"]
            g = self.est.p.get("residuos_por_tramo") or {}
            if len(g.get(t, ())) >= 12:
                q = g[t]
        return (float(cerrado + abierto * np.exp(self._qconf(q, a, False))),
                float(cerrado + abierto * np.exp(self._qconf(q, 1 - a, True))))

    # ------------------------------------------------------------------ publico
    def _medias(self, tareas):
        """cabeza B corregida por el estilo del generador: la base de todo el anclaje"""
        return np.asarray(self.est._media_tarea(tareas), float) * self.factor_estilo

    @staticmethod
    def _tareas_de(r):
        return [t for h in (r.get("historias") or []) for t in (h.get("tareas") or [])]

    def desplazamiento(self, r, horas):
        """El valor que hay que GUARDAR de un proyecto terminado, para mandarlo luego
        como previo_equipo (la media de los del equipo): log(horas reales / modelo).

        Se calcula aqui y no con desplazamiento() del estimador a proposito: el
        estimador lo mide contra su prediccion CRUDA, y este adaptador ancla contra la
        prediccion corregida por el estilo del generador (factor 0,7273 hoy). Un previo
        medido contra una base y aplicado sobre la otra sesgaria el total justo en ese
        factor. r: el backlog tal como se genero; horas: las reales, en el mismo orden
        que sus tareas."""
        tareas = self._tareas_de(r)
        if len(horas) != len(tareas):
            raise ValueError(f"horas tiene {len(horas)} valores y hay {len(tareas)} tareas")
        return float(np.log(np.sum(horas) / self._medias(tareas).sum()))

    def estimar(self, r, cerradas=None, respuestas=None, previo_equipo=None,
                peso_previo=1.0):
        """Anade horas a un resultado del generador. Devuelve el mismo diccionario.

        cerradas: horas reales ya registradas, en el mismo orden en que las tareas
        aparecen recorriendo las historias; None donde no las haya. Si se pasan, el
        total se ancla con ellas (cociente de sumas contraido con James-Stein).

        respuestas: lo que el usuario dice que tardo «la ultima vez» en algo asi,
        mismo orden, None donde no conteste. Anclan igual que una cerrada y la tarea
        cuenta por lo que dijo; si una tarea tiene las dos, gana la cerrada. No son
        horas registradas: no estrechan la banda.

        previo_equipo: media de desplazamiento() de los proyectos terminados del
        equipo. Es el punto de partida del anclaje, que las cerradas van corrigiendo;
        peso_previo lo atenua.

        Regla, la misma que estimar_proyecto del estimador:
            desplazamiento = (n * log(suma dada / suma modelo) + lambda * previo)
                             / (n + lambda),   n = cerradas + respuestas
        Sin respuestas y sin previo es exactamente la de antes.
        """
        if not self.disponible:
            r["estimacion"] = dict(disponible=False, motivo=self.motivo)
            return r
        historias = r.get("historias") or []
        # el INGLES, que es lo que el contrato del estimador pide
        tareas, tramos = [], []
        for h in historias:
            ts = h.get("tareas") or []
            tramos.append((len(tareas), len(tareas) + len(ts)))
            tareas.extend(ts)
        if not tareas:
            r["estimacion"] = dict(disponible=False, motivo="el backlog no tiene tareas")
            return r

        t0 = time.time()
        # cabeza B, corregida por el estilo del generador (ver FACTOR_ESTILO_TOTAL).
        # La cabeza A ya trae su propio factor aplicado dentro de estimar_tareas.
        medias = self._medias(tareas)
        # Sprint 0: plantillas escritas a mano, asi que NO llevan el factor de estilo del
        # generador sino el suyo propio (FACTOR_SPRINT0_*). Las listas de cerradas y
        # respuestas se refieren solo a las tareas de las historias.
        s0 = [h for h in (r.get("sprint0") or []) if h.get("tareas")]
        tareas_s0 = [x for h in s0 for x in h["tareas"]]
        medias_s0 = (np.asarray(self.est._media_tarea(tareas_s0), float) * FACTOR_SPRINT0_MEDIA
                     if tareas_s0 else np.zeros(0))
        t_media = time.time() - t0

        medianas = None
        if self.por_tarea:
            t1 = time.time()
            # estilo="generador": el factor de estilo de la cabeza A ya no se
            # aplica por omision. Se midio para ESTE texto, no para el que
            # escribe una persona, donde solo encogia la prediccion un 17 %.
            d = self.est.estimar_tareas(tareas, estilo="generador")  # cabeza A
            medianas = d.horas.to_numpy(float)
            probables = d.valor_probable.to_numpy(float)
            bajas = d.banda_baja.to_numpy(float)
            altas = d.banda_alta.to_numpy(float)
            t_mediana = time.time() - t1
        else:
            t_mediana = 0.
        if tareas_s0 and self.por_tarea:
            d0 = self.est.estimar_tareas(tareas_s0, estilo="persona")
            rej = np.asarray(self.est.p["rejilla"], float)
            h0 = d0.horas.to_numpy(float) * FACTOR_SPRINT0_MEDIANA
            p0 = rej[np.abs(np.log(rej)[None, :] - np.log(np.clip(h0, 1e-6, None))[:, None]).argmin(1)]
            b0 = d0.banda_baja.to_numpy(float) * FACTOR_SPRINT0_MEDIANA
            a0 = d0.banda_alta.to_numpy(float) * FACTOR_SPRINT0_MEDIANA
        i0 = 0
        for h in s0:
            j0 = i0 + len(h["tareas"])
            h["horas_suma"] = round(float(medias_s0[i0:j0].sum()), 1)
            if self.por_tarea:
                h["horas_tareas"] = [round(float(x), 2) for x in h0[i0:j0]]
                h["horas_probables"] = [round(float(x), 2) for x in p0[i0:j0]]
                h["horas_banda"] = [[round(float(x), 1), round(float(y), 1)]
                                    for x, y in zip(b0[i0:j0], a0[i0:j0])]
            i0 = j0

        for h, (a, b) in zip(historias, tramos):
            h["horas_suma"] = round(float(medias[a:b].sum()), 1)
            if medianas is not None:
                h["horas_tareas"] = [round(float(x), 2) for x in medianas[a:b]]
                h["horas_probables"] = [round(float(x), 2) for x in probables[a:b]]
                h["horas_banda"] = [[round(float(x), 1), round(float(y), 1)]
                                    for x, y in zip(bajas[a:b], altas[a:b])]

        n = len(tareas)

        def _lista(x, nombre):
            v = np.array([np.nan if c is None else float(c) for c in x], dtype=float)
            if len(v) != n:
                raise ValueError(f"{nombre} tiene {len(v)} valores y hay {n} tareas")
            return v

        cer = _lista(cerradas, "cerradas") if cerradas is not None else np.full(n, np.nan)
        real = ~np.isnan(cer)
        if respuestas is not None:
            cer = np.where(real, cer, _lista(respuestas, "respuestas"))  # gana la cerrada
        hay = ~np.isnan(cer)
        n_anclas = int(hay.sum())
        previo = peso_previo * float(previo_equipo) if previo_equipo is not None else 0.0
        off, respondido = 0.0, 0.0
        cerrado, abierto = 0.0, float(medias.sum()) + float(medias_s0.sum())
        total = abierto
        if n_anclas or previo:
            # El mismo anclaje que estimar_proyecto —cociente de sumas contraido con
            # James-Stein, ahora con el previo del equipo como punto de partida— pero
            # sobre las medias YA corregidas por estilo. No se delega en
            # estimar_proyecto porque volveria a predecir sin esa correccion, y la
            # contraccion hace que el factor no se cancele solo.
            lam = float(self.est.p.get("lambda", 12.))
            propio = np.log(cer[hay].sum() / medias[hay].sum()) if n_anclas else 0.0
            off = (n_anclas * propio + lam * previo) / (n_anclas + lam)
            cerrado = float(cer[real].sum())
            respondido = float(cer[hay & ~real].sum())
            abierto = respondido + float((medias[~hay] * np.exp(off)).sum())
            abierto += float(medias_s0.sum() * np.exp(off))
            total = cerrado + abierto
        # la banda se calibro con tareas cerradas de verdad: las respuestas no cuentan
        k = int(real.sum())
        baja, alta = self._banda(cerrado, abierto, len(tareas) + len(tareas_s0), k)
        r["estimacion"] = dict(
            disponible=True, total_horas=round(total, 1),
            banda_baja=round(baja, 1), banda_alta=round(alta, 1),
            nivel=self.nivel, tareas=len(tareas) + len(tareas_s0), cerradas=k,
            tareas_sprint0=len(tareas_s0),
            horas_sprint0=round(float(medias_s0.sum() * np.exp(off)), 1),
            respondidas=n_anclas - k, horas_respondidas=round(respondido, 1),
            previo_equipo=previo_equipo, desplazamiento=round(float(off), 4),
            por_tarea=bool(self.por_tarea),
            factor_estilo=round(self.factor_estilo, 4),
            fuente_estilo=self.fuente_estilo,
            dias_persona=round(total / 8, 1),
            segundos=round(time.time() - t0, 2),
            seg_total=round(t_media, 2), seg_tareas=round(t_mediana, 2))
        return r


def _demo():
    """Prueba de extremo a extremo: generar un backlog y estimarlo."""
    import importlib.util
    import json
    base = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("pl", base / "src" / "19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    libres = [a for a in sys.argv[1:] if not a.startswith("--")]
    desc = libres[0] if libres else (
        "Una aplicacion web para reservar servicios de limpieza a domicilio, "
        "con citas recurrentes y pagos en la app.")
    print(f"generando en {pl.DISPOSITIVO}...", flush=True)
    t0 = time.time()
    P = pl.Pipeline(traducir=True)
    r = P(desc)
    print(f"  backlog: {len(r['historias'])} historias · {r['n_tareas']} tareas · "
          f"{time.time() - t0:.1f} s", flush=True)
    e = Estimacion(por_tarea="--sin-tareas" not in sys.argv)
    print(f"  estimador cargado en {e.carga_s} s · disponible={e.disponible} "
          f"{e.motivo}", flush=True)
    r = e.estimar(r)
    est = r["estimacion"]
    if not est["disponible"]:
        print("  no se pudo estimar:", est["motivo"])
        return
    for h in r["historias"][:6]:
        print(f"\n  {h.get('historia_es') or h['historia']}")
        print(f"    subtotal {h['horas_suma']:.1f} h")
        for i, t_ in enumerate(h.get("tareas_es") or h["tareas"]):
            hh = f"{h['horas_tareas'][i]:5.2f} h" if h.get("horas_tareas") else "   ?  "
            print(f"      {hh}  {t_}")
    print(f"\n  TOTAL {est['total_horas']:.0f} h ({est['dias_persona']:.0f} dias-persona)"
          f" · banda al {est['nivel']:.0%} de {est['banda_baja']:.0f} a "
          f"{est['banda_alta']:.0f} h")
    print(f"  estimacion en {est['segundos']} s "
          f"(total {est['seg_total']} s + por tarea {est['seg_tareas']} s)")
    json.dump(r, open(base / "results" / "34_demo.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  escrito results/34_demo.json")


if __name__ == "__main__":
    _demo()
