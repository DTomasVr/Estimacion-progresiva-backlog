"""Taxonomia v4: los 40 sectores de v3 mas 11 nuevos.

Por que (plan de ruta, 07/11/2026): la prueba de extremo a extremo fallo en cuatro
dominios -limpieza, empleo, inventario y reserva de espacios- y la taxonomia no tenia
ninguno de los cuatro. Sin sector, el planificador de roles cae en el plantel del mas
parecido (limpieza -> enfermeras, salas -> instructores).

Que entra. Los cuatro que fallaron, obligatorios. Y de los otros 14 dominios que el
conjunto de calibracion tiene FUERA de la taxonomia, siete elegidos AL AZAR con semilla
(data/calibracion/reparto_dominios.json). Los otros siete se quedan fuera a proposito:
son el grupo de control. Comparando los dos grupos se sabe si anadir un sector mejora la
CALIDAD, o solo sube la similitud de sector (que subira siempre, por construccion), y la
puerta de entrada se sigue pudiendo probar con dominios desconocidos.

Como. Mismas plantillas y misma forma de componer que src/53_dominios_v3.py, leidas de su
codigo y no copiadas, para que los sectores nuevos tengan el estilo de los 40 que el
modelo ya conoce. Las funciones de cada sector NO repiten las del conjunto de calibracion. La primera
version si lo hacia sin querer: 12 de 72 funciones eran casi literales (SBERT >= 0,75)
y concentradas en los sectores tratados, lo que inflaba su similitud y contaminaba el
experimento. Se sustituyeron por otras funciones tipicas del dominio; cada sector sigue
siendo reconocible por su concepto central, que va en el nombre del producto.

La v3 no se toca. El pipeline lee v3 salvo que AGIX_TAXONOMIA=descripciones_v4.json.

out: data/synth/descripciones_v4.json  y  data/synth/roles_v4.json
"""
import ast
import collections
import itertools
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261107
random.seed(SEED)

# --- las plantillas de v3, leidas de su codigo ---------------------------------
_src = open(f"{BASE}/src/53_dominios_v3.py", encoding="utf-8").read()
_const = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(_src).body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
          and n.targets[0].id in ("PLANTILLAS", "PLANTILLAS_SIN", "VERBOS", "POR_SECTOR")}
PLANTILLAS, PLANTILLAS_SIN = _const["PLANTILLAS"], _const["PLANTILLAS_SIN"]
VERBOS, POR_SECTOR = _const["VERBOS"], _const["POR_SECTOR"]
assert len(PLANTILLAS) == len(PLANTILLAS_SIN) == 10


def plural(rol):                       # la misma que en v3
    if rol.endswith("staff"):
        return rol
    if rol.endswith("y") and rol[-2] not in "aeiou":
        return rol[:-1] + "ies"
    if rol.endswith(("s", "x", "ch", "sh")):
        return rol + "es"
    return rol + "s"


# --- los 11 sectores nuevos -----------------------------------------------------
# (articulo, funciones). Seis o siete funciones por sector, como en v3.
NUEVOS = {
    # obligatorios: los cuatro que fallaron
    # «housekeeping» es lo que escribe el traductor para «limpieza del hogar»: con solo
    # «home cleaning», la descripcion traducida seguia cayendo en alquiler de inmuebles
    "limpieza hogar": ("a home cleaning and housekeeping booking platform",
                       ["recurring cleaning plans", "hourly housekeeping visits", "cleaner availability by zone",
                        "visit checklists with photos", "ratings and tips",
                        "rescheduling and cancellations", "key handover notes",
                        "cleaning supply add-ons"]),
    "empleo": ("a job board",
               ["job postings with requirements", "candidate profiles with CV",
                "application stages", "interview scheduling", "skills matching",
                "offer letters", "placement statistics"]),
    "inventario": ("a multi-warehouse inventory system",
                   ["stock movements by warehouse", "inter-branch stock transfers",
                    "reorder points and low-stock warnings", "cycle counts",
                    "supplier purchase orders", "stock valuation reports"]),
    "reserva espacios": ("a campus space booking app",
                         ["recurring weekly reservations",
                          "availability calendar by building",
                          "capacity limits per room", "check-in at the room",
                          "no-show penalties", "equipment requests"]),
    # tratados, elegidos al azar
    "crm": ("a sales CRM",
            ["lead and opportunity tracking", "call and meeting logs",
             "sales pipeline by stage", "email templates for follow-ups",
             "quotes and proposals", "customer account history"]),
    "telemedicina": ("a telemedicine platform",
                     ["video consultations", "online appointment booking", "e-prescriptions",
                      "virtual waiting room", "consultation payments",
                      "follow-up reminders"]),
    "manufactura": ("a production control system",
                    ["shift scheduling for operators", "scrap and rework tracking",
                     "work station progress", "quality checks before dispatch",
                     "machine downtime logs", "batch traceability"]),
    "red vecinal": ("a neighbourhood community app",
                    ["community board posts", "lost and found posts",
                     "incident reports to the board", "visitor pre-registration",
                     "common area rules", "event announcements"]),
    "encuestas": ("a survey builder",
                  ["question editor with several question types", "anonymous responses",
                   "survey templates", "multi-language surveys", "conditional questions",
                   "response quotas and deadlines"]),
    "transporte escolar": ("a school bus tracking app",
                           ["live bus location", "boarding and drop-off alerts",
                            "route and stop planning", "driver assignment",
                            "absence notices", "vehicle maintenance log"]),
    "pedidos mayoristas": ("a wholesale ordering system",
                           ["next-day order cut-off", "standing weekly orders",
                            "delivery route sheets", "customer price lists",
                            "returns of unsold goods", "shop credit and invoices"]),
}
# Plantel por sector. Comprobados contra rol_valido() del pipeline el 07/11/2026; tres
# del borrador no pasaban y se cambiaron: dispatcher -> scheduling coordinator,
# lecturer -> teacher, respondent -> survey participant. «employer» se evita a proposito:
# en una sonda de 16 muestras el modelo lo obedece 11/16 y el resto escribe «lawyer».
ROLES = {
    "limpieza hogar": ["customer", "cleaner", "scheduling coordinator", "service manager"],
    "empleo": ["candidate", "recruiter", "hiring manager", "career advisor"],
    "inventario": ["warehouse clerk", "inventory manager", "branch manager", "purchasing officer"],
    "reserva espacios": ["student", "facilities officer", "teacher", "campus administrator"],
    "crm": ["sales rep", "sales manager", "account manager", "marketing analyst"],
    "telemedicina": ["patient", "doctor", "care coordinator", "clinic administrator"],
    "manufactura": ["plant manager", "machine operator", "quality inspector", "production planner"],
    "red vecinal": ["resident", "board member", "building manager", "moderator"],
    "encuestas": ["survey creator", "survey participant", "data analyst", "team admin"],
    "transporte escolar": ["parent", "bus driver", "transport coordinator", "school administrator"],
    "pedidos mayoristas": ["shop owner", "production planner", "delivery driver", "sales coordinator"],
}
assert set(NUEVOS) == set(ROLES)

# --- generar, con la misma composicion que v3 -------------------------------------
casos = []
for sector, (articulo, feats) in NUEVOS.items():
    pares = [(t, c) for t, c in itertools.product(range(len(PLANTILLAS)),
                                                  itertools.combinations(feats, 3))]
    random.shuffle(pares)
    roles = ROLES[sector]
    for i in range(POR_SECTOR):
        plantilla, (f1, f2, f3) = pares[i % len(pares)]
        elegidos = [roles[(i + j) % len(roles)] for j in range(3)]
        p1, p2, p3 = (plural(r) for r in elegidos)
        art = articulo[0].upper() + articulo[1:]
        verbo = random.choice(VERBOS)
        campos = dict(articulo=art, articulo_min=articulo, p1=p1, p2=p2, p3=p3,
                      verbo=verbo, f1=f1, f2=f2, f3=f3)
        casos.append(dict(id=f"v4-{sector}-{i}", sector=sector,
                          descripcion=PLANTILLAS[plantilla].format(**campos),
                          rasgos=[f1, f2, f3],
                          descripcion_sin=PLANTILLAS_SIN[plantilla].format(**campos),
                          roles=roles))

# --- quitar casi duplicadas, como en v3 (0,93 con el SBERT propio) -----------------
# Solo entre las nuevas: las de v3 ya pasaron su propio filtro.
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

torch.set_num_threads(int(os.environ.get("AGIX_HILOS", 2)))
sb = SentenceTransformer(f"{BASE}/models/sbert_agix_v1",
                         device=os.environ.get("AGIX_DEVICE", "cpu"))
E = sb.encode([c["descripcion"] for c in casos], normalize_embeddings=True,
              show_progress_bar=False)
S = E @ E.T
np.fill_diagonal(S, 0)
quedan, vistos = [], np.zeros(len(casos), bool)
for i in range(len(casos)):
    if vistos[i]:
        continue
    vistos |= (S[i] >= 0.93)
    quedan.append(casos[i])

v3 = json.load(open(f"{BASE}/data/synth/descripciones_v3.json", encoding="utf-8"))
roles_v3 = json.load(open(f"{BASE}/data/synth/roles_v3.json", encoding="utf-8"))
assert not set(NUEVOS) & set(roles_v3), "un sector nuevo ya existia en v3"
json.dump(v3 + quedan, open(f"{BASE}/data/synth/descripciones_v4.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
json.dump({**roles_v3, **ROLES}, open(f"{BASE}/data/synth/roles_v4.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)

c = collections.Counter(x["sector"] for x in quedan)
print(f"{len(casos)} generadas, {len(quedan)} tras quitar casi duplicadas")
print(f"v4: {len(v3) + len(quedan)} descripciones, {len(roles_v3) + len(ROLES)} sectores")
print(f"por sector nuevo: min {min(c.values())} max {max(c.values())}")
for s in NUEVOS:
    x = next(q for q in quedan if q["sector"] == s)
    print(f"  [{s}] {x['descripcion']}")
