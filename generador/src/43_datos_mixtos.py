"""Phase 3 of the distillation pilot: turn the LLM backlogs into training data
with exactly the same schema as the real data, and mix them.

Two datasets are written, matching the two halves:

  s2s_mixa_{train,val,test}   upper half: description -> ONE user story (One2One),
                              real o2ofmt + synthetic
  s2s_mix_{train,val,test}    lower half: story -> tasks, real hist + synthetic

40 of the descriptions are held out and never trained on. They become a
consumer-domain test set -- with the caveat that its references were written by
the teacher model, so it measures agreement with the teacher, not ground truth.

out: data/processed/s2s_mix*_{train,val,test}.parquet
"""
import collections
import json
import os
import re

import numpy as np
import pandas as pd
import sentencepiece as spm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260922
MAX_IN, MAX_OUT, EOS = 256, 192, 1
N_RESERVADOS = 40      # test
N_VAL = 25             # validacion, disjunto del test
PREF_TAREAS = "decompose story into tasks: "
PREF_ARRIBA = "feature -> user story: "
FORMA = re.compile(r"^as\s+(?:an?|the)\s+[^,]{2,60},?\s+i\s+want\b.*", re.I)
MODELO = "qwen2.5_7b"
# la tanda v3 del profesor usa exactamente este mismo codigo de mezcla, solo
# cambia el fichero de entrada y el prefijo de salida:
#   PBE_SYNTH=backlogs_v3_qwen2.5_7b.jsonl  PBE_PREF=v3  python src/43_datos_mixtos.py
SYNTH = os.environ.get("PBE_SYNTH", f"backlogs_{MODELO}.jsonl")
# 0 = sin tope (como antes); 2 = como maximo dos tareas por arranque y backlog
TOPE_ARRANQUE = int(os.environ.get("PBE_TOPE_ARRANQUE", 0))
PREF = os.environ.get("PBE_PREF", "mix")

sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
sep = sp.encode(";")


def objetivo_ids(tareas):
    ids = []
    for k, t in enumerate(tareas):
        extra = (sep if k else []) + sp.encode(t)
        if len(ids) + len(extra) + 1 > MAX_OUT:
            break
        ids += extra
    return ids + [EOS]


def entrada_ids(src):
    return sp.encode(src)[:MAX_IN - 1] + [EOS]


# ------------------------------------------------------------------ leer LLM
ruta = f"{BASE}/data/synth/{SYNTH}"
B = [json.loads(l) for l in open(ruta, encoding="utf-8")]
print(f"{len(B)} backlogs del profesor, {len({b['sector'] for b in B})} sectores")

rng = np.random.default_rng(SEED)
ids = sorted({b["id"] for b in B})
sorteo = rng.permutation(ids)
reservados = set(sorteo[:N_RESERVADOS].tolist())              # test
val_ids = set(sorteo[N_RESERVADOS:N_RESERVADOS + N_VAL].tolist())   # validacion, disjunto
assert not (reservados & val_ids)
print(f"{len(reservados)} descripciones para test y {len(val_ids)} para validacion, "
      f"disjuntas y nunca entrenadas")

filas_arriba, filas_abajo = [], []
vistas_h, vistas_t = set(), set()
for b in B:
    arranques_bl = collections.Counter()
    split = ("test" if b["id"] in reservados else "val" if b["id"] in val_ids else "train")
    for h in b["historias"]:
        hist = h["historia"].strip()
        if not FORMA.match(hist) or len(hist) < 25:
            continue                                   # solo historias bien formadas
        k = re.sub(r"[^a-z0-9]+", " ", hist.lower()).strip()
        if k in vistas_h:
            continue                                   # duplicadas entre sectores
        vistas_h.add(k)
        filas_arriba.append(dict(entrada=b["descripcion"], objetivo=hist, estilo="sintetico",
                                 proj=f"synth:{b['sector']}", split=split, parent_uid=b["id"],
                                 src=f"{PREF_ARRIBA}{b['descripcion']} | examples: "))
        tareas = []
        for t in h["tareas"]:
            # Tope de arranques repetidos DENTRO del backlog. Medido: el profesor
            # concentra el 28 % de sus tareas en cinco arranques de dos palabras y
            # el alumno lo amplifica al 83-87 %, cuando las tareas escritas por
            # personas en el corpus de Jira estan en el 2,6 %. Quitar las
            # repetidas en origen es gratis: no hay que regenerar nada.
            if TOPE_ARRANQUE:
                w = re.findall(r"[a-z']+", t.lower())[:2]
                k2 = " ".join(w) if len(w) == 2 else None
                if k2 and arranques_bl[k2] >= TOPE_ARRANQUE:
                    continue
                if k2:
                    arranques_bl[k2] += 1
            t = t.strip().replace(";", ",")
            kt = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
            if len(t) >= 8 and kt not in {re.sub(r"[^a-z0-9]+", " ", x.lower()).strip() for x in tareas}:
                tareas.append(t)
        if len(tareas) >= 2:
            filas_abajo.append(dict(mitad="tasks", estilo="sintetico", proj=f"synth:{b['sector']}",
                                    split=split, parent_uid=f"{b['id']}#{len(filas_abajo)}",
                                    src=f"{PREF_TAREAS}{hist} | examples: ", tareas=tareas))

# 40 % of the teacher's stories have a near-duplicate, almost all inside the same
# sector (22 descriptions per sector share features). Redundant rows inflate the
# set without teaching anything, so collapse them: hard inside one backlog, looser
# across the corpus, where a repeated phrasing under a DIFFERENT description is
# still a useful example.
from sentence_transformers import SentenceTransformer  # noqa: E402

_disp = "cuda" if os.environ.get("PBE_SBERT_GPU") == "1" else "cpu"
_sb = SentenceTransformer(f"{BASE}/models/sbert_pbe_v1", device=_disp)
_txt = [f["objetivo"] for f in filas_arriba]
_E = _sb.encode(_txt, convert_to_tensor=True, normalize_embeddings=True,
                batch_size=128, show_progress_bar=False)
_S = (_E @ _E.T).float().cpu().numpy()
np.fill_diagonal(_S, 0)
_uid = np.array([f["parent_uid"] for f in filas_arriba])
_mismo = _uid[:, None] == _uid[None, :]
_dup = np.where(_mismo, _S >= 0.90, _S >= 0.95)     # mas duro dentro del mismo backlog
_vistos = np.zeros(len(filas_arriba), bool)
_fuera = set()
for i in range(len(filas_arriba)):
    if _vistos[i]:
        continue
    nuevos = _dup[i] & ~_vistos
    _fuera.update(np.flatnonzero(nuevos).tolist())
    _vistos |= nuevos
    _vistos[i] = True
print(f"casi duplicadas quitadas: {len(_fuera)} de {len(filas_arriba)} "
      f"({100*len(_fuera)/max(1,len(filas_arriba)):.1f} %)")
_claves = {filas_arriba[i]["objetivo"] for i in _fuera}
filas_arriba = [f for i, f in enumerate(filas_arriba) if i not in _fuera]
filas_abajo = [f for f in filas_abajo
               if f["src"][len(PREF_TAREAS):].split(" | examples:")[0] not in _claves]

# --- variante de la descripcion SIN nombrar al publico (solo tanda v3) -------
# Las descripciones que llegan a PBE muchas veces no dicen para quien es el
# producto. Si todos los ejemplos lo dicen, la mitad de arriba aprende a copiar
# un sustantivo de la entrada al hueco del rol ("As a service appointment, ...")
# en lugar de elegir un actor plausible del dominio. Se duplica cada ejemplo con
# la entrada despublicada y el MISMO objetivo, para ensenar esa invariancia.
# Va DESPUES del deduplicador a proposito: el objetivo es identico y lo borraria.
_sinpub = {}
_dv3 = f"{BASE}/data/synth/descripciones_v3.json"
if os.path.exists(_dv3):
    _sinpub = {x["id"]: x.get("descripcion_sin") for x in
               json.load(open(_dv3, encoding="utf-8")) if x.get("descripcion_sin")}
_n_sin_gemelas = len(filas_arriba)
_gemelas = []
for f in filas_arriba:
    s = _sinpub.get(f["parent_uid"])
    if s and s != f["entrada"]:
        g = dict(f)
        g["entrada"] = s
        g["src"] = f"{PREF_ARRIBA}{s} | examples: "
        g["estilo"] = "sintetico_sin_publico"
        _gemelas.append(g)
if _gemelas:
    print(f"variante sin publico: +{len(_gemelas)} pares de arriba "
          f"({len(filas_arriba)} -> {len(filas_arriba)+len(_gemelas)})")
    filas_arriba += _gemelas

# --- variante con el ROL como ficha de control (solo tanda v3) ---------------
# Segundo metodo, para comparar contra el filtrado: en lugar de esperar que el
# modelo elija un rol valido y descartar cuando no lo hace, se le DICE cual.
#   feature -> user story | role: shopper | <descripcion> | examples:
# La ficha va antes de la descripcion para que no la corte el truncado. Se
# entrena con la descripcion completa y con la despublicada, para que el modelo
# aprenda a usar la ficha y no a copiar un sustantivo de la entrada.
# Aviso del historial: la ficha de estilo "(story form)" fracaso con 195
# ejemplos; aqui hay ~7.000, que es otro regimen.
_ROL = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*,", re.I)
filas_rol = []
for f in filas_arriba[:_n_sin_gemelas]:      # sin las gemelas: el bucle de abajo
                                             # ya genera las dos entradas y si no
                                             # saldrian duplicados exactos
    m = _ROL.match(f["objetivo"])
    if not m:
        continue
    rol = re.sub(r"\s+", " ", m.group(1).strip().lower())
    for entrada in {f["entrada"], _sinpub.get(f["parent_uid"]) or f["entrada"]}:
        g = dict(f)
        g["entrada"] = entrada
        g["src"] = f"feature -> user story | role: {rol} | {entrada} | examples: "
        filas_rol.append(g)
if filas_rol:
    print(f"variante con ficha de rol: {len(filas_rol)} pares de arriba")

# --- paso 2: ficha de rol MAS ficha de funcion -------------------------------
#   feature -> user story | role: shopper | funcion: view | <descripcion>
# La clase de funcion sale de agrupar los verbos del medio con el SBERT propio
# (src/84_clases_funcion.py), no de una lista escrita a mano. Motivo medido: la
# salida del modelo se concentra en la clase "view" (42 % contra 22 % de las
# historias humanas y del profesor) y casi no produce "manage" (2 % contra 6-11 %),
# o sea que el backlog sale lleno de "quiero ver X". La entropia de clases es 2,22
# contra 2,45 humana y 2,51 del profesor, sobre un maximo de 2,58.
_CLASES = {}
_pc = f"{BASE}/data/processed/clases_funcion.json"
if os.path.exists(_pc):
    _CLASES = json.load(open(_pc, encoding="utf-8")).get("verbo_clase", {})
_PARADA_V = {"the", "a", "an", "my", "their", "all", "some", "this", "that", "to",
             "be", "have", "get", "make", "do", "able"}


def _clase_funcion(objetivo):
    """clase de la funcion que pide la historia, por el primer verbo de su medio"""
    m = re.match(r"^\s*as\s+[^,]+,\s*i\s*(?:want|need|wish|would like|am able|can)"
                 r"\s*(?:to\s+|a\s+|able\s+to\s+)?(.*)$", objetivo or "", re.I)
    if not m:
        return None
    for w in re.findall(r"[a-z]+", m.group(1).lower()):
        if w not in _PARADA_V:
            return _CLASES.get(w)
    return None


filas_rolfun = []
if _CLASES:
    for f in filas_rol:
        cl = _clase_funcion(f["objetivo"])
        if not cl:
            continue
        g = dict(f)
        # la ficha de funcion va DESPUES de la de rol, y el prompt de inferencia
        # tiene que usar el mismo orden o el modelo no la reconoce
        g["src"] = re.sub(r"(\| role: [^|]+\| )", lambda m: m.group(1) + f"funcion: {cl} | ",
                          f["src"], count=1)
        filas_rolfun.append(g)
    print(f"variante con ficha de rol y funcion: {len(filas_rolfun)} pares de arriba")

A = pd.DataFrame(filas_arriba)
L = pd.DataFrame(filas_abajo)
A["input_ids"] = [entrada_ids(s) for s in A.src]
A["labels"] = [objetivo_ids([o]) for o in A.objetivo]
L["input_ids"] = [entrada_ids(s) for s in L.src]
L["labels"] = [objetivo_ids(list(t)) for t in L.tareas]
print(f"sinteticos utiles: {len(A)} pares descripcion->historia, {len(L)} historia->tareas "
      f"({np.mean([len(t) for t in L.tareas]):.1f} tareas por historia)")

# --------------------------------------------------------------- mezclar
for etiqueta, sint, reales, cols in [
    (f"{PREF}a", A, {s: f"{BASE}/data/processed/s2s_o2ofmt_{s}.parquet" for s in ["train", "val", "test"]},
     ["src", "input_ids", "labels", "estilo", "split", "parent_uid", "proj"]),
    (PREF, L, {s: f"{BASE}/data/processed/s2s_hist_{s}.parquet" for s in ["train", "val", "test"]},
     ["src", "input_ids", "labels", "estilo", "split", "parent_uid", "proj", "mitad", "tareas"]),
]:
    for split in ["train", "val", "test"]:
        R = pd.read_parquet(reales[split])
        for c in cols:
            if c not in R.columns:
                R[c] = None
            if c not in sint.columns:
                sint[c] = None
        S = sint[sint.split == split][cols]   # cada rebanada de su propio grupo, sin cruzarse
        D = pd.concat([R[cols], S], ignore_index=True)
        D.to_parquet(f"{BASE}/data/processed/s2s_{etiqueta}_{split}.parquet", index=False)
        print(f"  s2s_{etiqueta}_{split}: {len(R)} reales + {len(S)} sinteticos = {len(D)}")

# --- conjunto con ficha de rol: <PREF>r --------------------------------------
# Los reales no llevan ficha (no se conoce el rol de una epica), asi que se
# concatenan tal cual: el modelo ve las dos formas y aprende que la ficha, si
# esta, manda.
if filas_rol:
    Ar = pd.DataFrame(filas_rol)
    Ar["input_ids"] = [entrada_ids(s) for s in Ar.src]
    Ar["labels"] = [objetivo_ids([o]) for o in Ar.objetivo]
    cols = ["src", "input_ids", "labels", "estilo", "split", "parent_uid", "proj"]
    for split in ["train", "val", "test"]:
        R = pd.read_parquet(f"{BASE}/data/processed/s2s_o2ofmt_{split}.parquet")
        for c in cols:
            if c not in R.columns:
                R[c] = None
            if c not in Ar.columns:
                Ar[c] = None
        # con ficha Y sin ficha: el planificador devuelve vacio en dominios que
        # no reconoce, y entonces la entrada llega sin ficha. Si el modelo nunca
        # hubiera visto estos dominios sin ficha, ahi solo tendria de donde
        # copiar los ejemplos de Jira, que es de donde venia el defecto.
        for c in cols:
            if c not in A.columns:
                A[c] = None
        S = pd.concat([Ar[Ar.split == split][cols], A[A.split == split][cols]],
                      ignore_index=True)
        D = pd.concat([R[cols], S], ignore_index=True)
        D.to_parquet(f"{BASE}/data/processed/s2s_{PREF}r_{split}.parquet", index=False)
        print(f"  s2s_{PREF}r_{split}: {len(R)} reales + {len(S)} con ficha = {len(D)}")

# --- conjunto con las DOS fichas: <PREF>f ------------------------------------
if filas_rolfun:
    Af = pd.DataFrame(filas_rolfun)
    Af["input_ids"] = [entrada_ids(s_) for s_ in Af.src]
    Af["labels"] = [objetivo_ids([o]) for o in Af.objetivo]
    cols = ["src", "input_ids", "labels", "estilo", "split", "parent_uid", "proj"]
    for split in ["train", "val", "test"]:
        R = pd.read_parquet(f"{BASE}/data/processed/s2s_o2ofmt_{split}.parquet")
        for c in cols:
            if c not in R.columns:
                R[c] = None
            if c not in Af.columns:
                Af[c] = None
        S = Af[Af.split == split][cols]
        D = pd.concat([R[cols], S], ignore_index=True)
        D.to_parquet(f"{BASE}/data/processed/s2s_{PREF}f_{split}.parquet", index=False)
        print(f"  s2s_{PREF}f_{split}: {len(R)} reales + {len(S)} con dos fichas = {len(D)}")
    print(" EJEMPLO con ficha:", Ar.src.iloc[0][:170])

print("\nejemplo sintetico de arriba:")
print(" SRC:", A.src.iloc[0][:150])
print(" TGT:", A.objetivo.iloc[0][:150])
print("ejemplo sintetico de abajo:")
print(" SRC:", L.src.iloc[0][:150])
print(" TGT:", " ; ".join(L.tareas.iloc[0])[:190])
