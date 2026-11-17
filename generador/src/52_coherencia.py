"""Mide dos defectos concretos que la metrica sF1 no ve:

  1. DIVERSIDAD DE ROLES  -- cuantos roles distintos aparecen en un backlog.
     Si todas las historias dicen "As a user", el backlog no representa al
     equipo real y el modelo aprende a repetir un rol.

  2. ATRIBUCION DE TAREAS -- dada una tarea, ¿su propia historia es la mas
     parecida de todo el backlog? Si las tareas son genericas ("implement the
     backend endpoint") daran igual con cualquier historia, y eso es
     exactamente la incoherencia que se ve al leer las salidas.
     Azar = 1/n_historias.

  python src/52_coherencia.py profesor        # datos del profesor
  python src/52_coherencia.py modelo          # salida del pipeline en produccion
  python src/52_coherencia.py ambos
"""
import collections
import importlib.util
import json
import os
import re
import statistics as st
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SBERT = f"{BASE}/models/sbert_agix_v1"

RE_ROL = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*,", re.I)


def rol(h):
    m = RE_ROL.match(h)
    return m.group(1).strip().lower() if m else None


def cargar_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT, device=os.environ.get("AGIX_DEVICE", "cuda"))


def atribucion(modelo, backlogs):
    """backlogs = lista de listas [(historia, [tareas...]), ...]"""
    aciertos = tot = 0
    margenes = []
    por_backlog = []
    azar = []
    for bl in backlogs:
        hs = [h for h, ts in bl if ts]
        if len(hs) < 4:
            continue
        idx, tareas = [], []
        for i, (h, ts) in enumerate([(h, t) for h, t in bl if t]):
            for t in ts:
                idx.append(i)
                tareas.append(t)
        if not tareas:
            continue
        E_h = modelo.encode(hs, normalize_embeddings=True, show_progress_bar=False)
        E_t = modelo.encode(tareas, normalize_embeddings=True, show_progress_bar=False)
        S = E_t @ E_h.T                                   # tareas x historias
        a = 0
        for j, i in enumerate(idx):
            prop = S[j, i]
            otras = np.delete(S[j], i)
            if prop >= otras.max():
                a += 1
            margenes.append(float(prop - otras.max()))
        aciertos += a
        tot += len(idx)
        por_backlog.append(a / len(idx))
        azar.append(1 / len(hs))
    return dict(acc=aciertos / tot, n=tot, margen=st.mean(margenes),
                margen_med=st.median(margenes), por_backlog=st.mean(por_backlog),
                azar=st.mean(azar))


# La atribucion mide si una tarea se parece mas a SU historia que a las
# hermanas, pero no ve los dos defectos que mas molestan al leer: el relleno de
# plantilla y el eco. Estas dos medidas son literales y no usan modelo.
RELLENO = [
    re.compile(r"^implement a feature to", re.I),
    re.compile(r"^create a dashboard to (display|show|view)", re.I),
    re.compile(r"^develop a .{0,30}tracking feature", re.I),
    re.compile(r"filter .{0,30}by date range", re.I),
    re.compile(r"^(add|create|implement) (a )?(feature|functionality)\b", re.I),
]
PARADA = {"a", "an", "the", "to", "for", "of", "in", "on", "and", "with", "by",
          "from", "at", "as", "their", "its", "my", "all", "new", "each", "per",
          "be", "is", "are", "should", "able", "that", "this", "it"}


RE_QUIERO = re.compile(r"i\s*(?:'d like|would like|want|need)\s+(?:to\s+)?([a-z]+(?:\s+[a-z]+)?)", re.I)
RE_PARAQUE = re.compile(r"so\s+that\s+(.{0,40})", re.I)


def historia_info(backlogs):
    """Plantilla a nivel de HISTORIA, no de tarea. Referencia medida sobre las
    328 historias humanas: verbo tras 'I want' 34,0 %, clausula 'so that' 10,4 %.
    Sirve para separar dos cosas que se confunden: que las TAREAS colapsen en
    plantilla (nos pasa, 84 %) y que las HISTORIAS lo hagan (no nos pasa)."""
    v, s = [], []
    for bl in backlogs:
        for h, _ts in bl:
            m = RE_QUIERO.search(h)
            if m:
                v.append(m.group(1).lower())
            m = RE_PARAQUE.search(h)
            if m:
                s.append(re.sub(r"[^a-z ]", "", m.group(1).lower()).strip()[:22])

    def top5(xs):
        c = collections.Counter(xs)
        t = sum(c.values())
        return (sum(n for _, n in c.most_common(5)) / max(1, t), [k for k, _ in c.most_common(5)])

    cv, tv = top5(v)
    cs, _ = top5(s)
    return dict(verbo=cv, verbo_top=tv, paraque=cs, n=len(v))


_LEX_JIRA = None


def fugas_jira(backlogs):
    """Proporcion de tareas que contienen vocabulario propio del corpus de Jira.
    El lexico se deriva por frecuencias en src/70_lexico_jira.py, no a mano.
    Controles medidos sobre los datos de entrenamiento: las tareas de Jira dan
    72,7 % y las sinteticas 2,4 %."""
    global _LEX_JIRA
    if _LEX_JIRA is None:
        p = f"{BASE}/data/processed/lexico_jira.json"
        _LEX_JIRA = set(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else set()
    if not _LEX_JIRA:
        return None
    n = con = 0
    for bl in backlogs:
        for _h, ts in bl:
            for t in ts:
                n += 1
                if _LEX_JIRA & set(re.findall(r"[a-z]{4,}", t.lower())):
                    con += 1
    return con / max(1, n)


def texto_info(backlogs):
    # Medir el relleno con una lista fija de frases fue un error: la tanda v3
    # no quito la plantilla, la CAMBIO ("Implement a feature to X" ->
    # "Store X in the database"), y la lista vieja daba 0 %. Asi que se mide la
    # CONCENTRACION: que parte de las tareas empieza por uno de los cinco
    # arranques de dos palabras mas frecuentes. Eso detecta cualquier plantilla,
    # vieja o nueva, sin tener que anticiparla.
    arranques = collections.Counter()
    for bl in backlogs:
        for _h, ts in bl:
            for t in ts:
                w = re.findall(r"[a-z']+", t.lower())[:2]
                if len(w) == 2:
                    arranques[" ".join(w)] += 1
    total_arr = sum(arranques.values())
    top5 = [k for k, _ in arranques.most_common(5)]
    conc = sum(arranques[k] for k in top5) / max(1, total_arr)

    n = relleno = eco = anclada = 0
    for bl in backlogs:
        for h, ts in bl:
            hp = set(re.findall(r"[a-z]+", h.lower())) - PARADA
            for t in ts:
                n += 1
                if any(p.search(t) for p in RELLENO):
                    relleno += 1
                p = [w for w in re.findall(r"[a-z]+", t.lower()) if w not in PARADA]
                if len(p) <= 9 and len(p) != len(set(p)):
                    eco += 1
                # anclada = nombra algo concreto de SU historia y ademas aporta
                # palabras propias, o sea no es ni generica ni un eco
                if p and (set(p) & hp) and len(set(p) - hp) / len(set(p)) >= 0.3:
                    anclada += 1
    return dict(n=n, relleno=relleno / n, eco=eco / n, anclada=anclada / n,
                conc=conc, top5=top5)


def roles_info(backlogs):
    nr, dom, sin = [], [], 0
    tot = 0
    cont = collections.Counter()
    for bl in backlogs:
        rs = []
        for h, ts in bl:
            tot += 1
            r = rol(h)
            if r is None:
                sin += 1
            else:
                rs.append(r)
                cont[r] += 1
        if rs:
            nr.append(len(set(rs)))
            dom.append(collections.Counter(rs).most_common(1)[0][1] / len(rs))
    return dict(roles_med=st.median(nr), roles_media=sum(nr) / len(nr),
                un_solo_rol=sum(1 for x in nr if x == 1) / len(nr),
                cuota_dominante=sum(dom) / len(dom), sin_patron=sin / tot,
                distintos=len(cont), top=cont.most_common(8))


def informe(nombre, backlogs, modelo):
    print(f"\n{'='*78}\n  {nombre}   ({len(backlogs)} backlogs)\n{'='*78}")
    r = roles_info(backlogs)
    print(f"ROLES   un solo rol en el backlog : {100*r['un_solo_rol']:5.1f} %  "
          f"(mediana {r['roles_med']:.0f}, media {r['roles_media']:.2f} roles)")
    print(f"        cuota del rol dominante   : {100*r['cuota_dominante']:5.1f} %")
    print(f"        sin patron 'As a X,'      : {100*r['sin_patron']:5.1f} %")
    print(f"        roles distintos en total  : {r['distintos']}")
    print(f"        top: {', '.join(f'{k} ({v})' for k, v in r['top'])}")
    hi = historia_info(backlogs)
    print(f"HIST.   verbo tras 'I want'        : {100*hi['verbo']:5.1f} %  (humanos 34,0 %)  "
          f"[{', '.join(hi['verbo_top'][:4])}]")
    print(f"        clausula 'so that'         : {100*hi['paraque']:5.1f} %  (humanos 10,4 %)")
    f = fugas_jira(backlogs)
    if f is not None:
        print(f"FUGAS   vocabulario del corpus de Jira : {100*f:5.1f} %  "
              f"(sintetico 2,4 %, Jira 72,7 %)")
    t = texto_info(backlogs)
    print(f"TAREAS  relleno de plantilla        : {100*t['relleno']:5.1f} %  (n={t['n']})")
    print(f"        concentracion top5 arranques: {100*t['conc']:5.1f} %  "
          f"[{' | '.join(t['top5'])}]")
    print(f"        eco (repite un sustantivo)  : {100*t['eco']:5.1f} %")
    print(f"        ancladas a SU historia      : {100*t['anclada']:5.1f} %")
    a = atribucion(modelo, backlogs)
    print(f"ATRIB.  acierto (tarea -> su historia) : {100*a['acc']:5.1f} %   "
          f"(azar {100*a['azar']:.1f} %, n={a['n']} tareas)")
    print(f"        margen propio - mejor ajena    : {a['margen']:+.4f} (mediana {a['margen_med']:+.4f})")
    return r, a, t


def prof(fichero="backlogs_qwen2.5_7b.jsonl", solo_v1=True):
    L = [json.loads(l) for l in open(f"{BASE}/data/synth/{fichero}", encoding="utf-8")]
    if solo_v1:      # la tanda v1 quedo mezclada con la v2 en el mismo fichero
        sec = {x["sector"] for x in
               json.load(open(f"{BASE}/data/synth/descripciones_v1.json", encoding="utf-8"))}
        L = [b for b in L if b["sector"] in sec]
    return [[(h["historia"], list(h["tareas"])) for h in b["historias"]] for b in L]


DESCRIPCIONES = [
    "A supermarket app where customers browse the catalogue, build a basket and book a delivery slot.",
    "An online store for selling audiobooks and ebooks, with a library and offline listening.",
    "A gym management app with memberships, class booking and a check-in kiosk.",
    "A veterinary clinic system for appointments, vaccination records and reminders.",
    "A hotel booking platform with room availability, seasonal rates and guest reviews.",
    "A food delivery app connecting restaurants, couriers and customers with live tracking.",
    "A school platform where teachers publish assignments and parents follow grades.",
    "A car workshop system for service appointments, repair quotes and spare part stock.",
    "A recycling collection service with household pickup scheduling and weight reports.",
    "A microloan platform where borrowers apply and analysts review credit risk.",
]


def modelo_salida(k=12):
    os.environ.setdefault("AGIX_DEVICE", "cuda")
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    if os.environ.get("AGIX_ARRIBA"):        # medir otra version sin editar nada
        pl.M_ARRIBA = os.environ["AGIX_ARRIBA"]
    if os.environ.get("AGIX_ABAJO"):
        pl.M_ABAJO = os.environ["AGIX_ABAJO"]
        pl.M_ABAJO2 = os.environ.get("AGIX_ABAJO2", pl.M_ABAJO2)
    print(f"  arriba: {os.path.basename(pl.M_ARRIBA)}\n  abajo : {os.path.basename(pl.M_ABAJO)}")
    P = pl.Pipeline(traducir=False)
    out = []
    for d in DESCRIPCIONES:
        r = P(d, k_historias=k, n_muestras=36)
        out.append([(h["historia"], list(h["tareas"])) for h in r["historias"]])
        print(f"  generado: {d[:60]}...  {len(out[-1])} historias")
    return out


if __name__ == "__main__":
    cual = sys.argv[1] if len(sys.argv) > 1 else "ambos"
    m = cargar_sbert()
    if cual in ("profesor", "ambos"):
        informe("PROFESOR v1 - 30 sectores", prof(), m)
    if cual == "profesor_v3":
        informe("PROFESOR v3 - 40 sectores, plantel de roles",
                prof("backlogs_v3_qwen2.5_7b.jsonl", solo_v1=False), m)
    if cual in ("modelo", "ambos"):
        informe("MODELO", modelo_salida(), m)
