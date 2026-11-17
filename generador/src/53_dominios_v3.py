"""Descripciones v3. Corrige dos defectos medidos en la version v1/v2:

 1. ROLES. En v1 cada descripcion nombraba UN solo publico
    (`publico=random.choice(publicos)`), asi que el profesor escribia las 12
    historias para ese unico rol: el 63,5 % de los backlogs tenia un solo rol.
    El modelo aprendio "elige un sustantivo y repitelo", y en produccion
    rellena el hueco del rol con cualquier sustantivo de la descripcion
    ("As a service appointment, I want ...").
    Aqui cada descripcion nombra 2 o 3 publicos, y ademas se escribe un
    PLANTEL DE ACTORES por sector que va dentro del prompt del profesor.

 2. COBERTURA. Se anaden 10 sectores de consumo que faltaban, manteniendo la
    PROFUNDIDAD por sector (~20 descripciones). La leccion de v2 fue que
    ensanchar quitando profundidad empeora: 211 sectores con 6 descripciones
    cada uno no gano a 30 sectores con 18.

out: data/synth/descripciones_v3.json  y  data/synth/roles_v3.json
"""
import ast
import collections
import itertools
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260926
random.seed(SEED)

# se reutiliza la taxonomia de v1 sin copiarla, para que no se desincronicen
_src = open(f"{BASE}/src/41_dominios_sinteticos.py", encoding="utf-8").read()
SECTORES = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SECTORES")

# --- 10 sectores de consumo que faltaban ------------------------------------
NUEVOS = {
    "supermercado": ("an online supermarket", ["shoppers", "pickers", "store managers"],
                     ["catalogue by aisle", "basket and substitutions", "delivery slot booking",
                      "weight-based products", "loyalty discounts", "stock shortages",
                      "recurring shopping lists"]),
    "farmacia": ("a pharmacy app", ["patients", "pharmacists", "couriers"],
                 ["prescription upload", "medicine search with equivalents", "home delivery",
                  "refill reminders", "controlled drug validation", "insurance coverage check"]),
    "belleza": ("a salon booking app", ["clients", "stylists", "salon owners"],
                ["service catalogue with duration", "stylist agenda", "deposits for no-shows",
                 "loyalty stamps", "before and after photos", "product sales at checkout"]),
    "parqueo": ("a street parking app", ["drivers", "parking inspectors", "city operators"],
                ["zone rates by hour", "session start and stop", "expiry reminders",
                 "monthly invoices", "fine appeals", "resident permits"]),
    "lavanderia": ("a laundry pickup service", ["customers", "drivers", "plant operators"],
                   ["pickup and drop-off windows", "garment counting", "stain treatment notes",
                    "per-kilo and per-item pricing", "delay compensation", "subscription plans"]),
    "cine": ("a cinema ticketing app", ["moviegoers", "cinema managers"],
             ["showtimes by cinema", "seat map selection", "combo snacks preorder",
              "loyalty points", "refunds for cancelled shows", "age rating checks"]),
    "movilidad": ("a scooter sharing app", ["riders", "field technicians", "city operators"],
                  ["vehicle map with battery level", "unlock by QR", "ride pricing per minute",
                   "parking zone rules", "damage reports", "battery swap routes"]),
    "colegio": ("a school administration platform", ["parents", "teachers", "principals"],
                ["enrolment and tuition", "attendance records", "report cards",
                 "parent-teacher meetings", "school bus tracking", "canteen balance"]),
    "banca": ("a digital bank app", ["account holders", "support agents", "compliance officers"],
              ["account opening with identity check", "transfers and scheduled payments",
               "card freezing and limits", "transaction disputes", "statements and tax documents",
               "fraud alerts"]),
    "telecom": ("a mobile operator self-service app", ["subscribers", "support agents"],
                ["data usage meter", "plan changes and add-ons", "top-ups",
                 "roaming activation", "device financing", "coverage incident reports"]),
}
SECTORES.update(NUEVOS)

# --- plantel de actores por sector ------------------------------------------
# En singular y en forma de persona: es lo que ira LITERALMENTE dentro del
# prompt, para que el profesor reparta las historias entre actores validos en
# lugar de inventar el hueco del rol.
ROLES = {
    "e-commerce": ["shopper", "seller", "store manager", "warehouse picker", "support agent"],
    "marketplace": ["buyer", "independent seller", "moderator", "category manager", "support agent"],
    "libros y audio": ["reader", "publisher", "narrator", "catalogue editor"],
    "streaming": ["viewer", "content editor", "parent", "rights manager"],
    "delivery": ["customer", "courier", "restaurant owner", "operations manager"],
    "turismo": ["traveller", "hotel manager", "front desk agent", "revenue manager"],
    "salud": ["patient", "doctor", "receptionist", "clinic manager", "lab technician"],
    "fitness": ["member", "trainer", "gym manager", "front desk staff"],
    "educacion": ["student", "instructor", "coordinator", "teaching assistant"],
    "finanzas": ["account holder", "support agent", "risk analyst"],
    "seguros": ["policyholder", "broker", "claim adjuster", "underwriter"],
    "inmobiliaria": ["property buyer", "real estate agent", "landlord", "mortgage adviser"],
    "rrhh": ["candidate", "recruiter", "hiring manager", "HR administrator"],
    "eventos": ["attendee", "event organiser", "promoter", "door staff"],
    "logistica": ["sender", "warehouse operator", "driver", "logistics coordinator"],
    "restaurantes": ["waiter", "cook", "restaurant owner", "cashier"],
    "inmuebles alquiler": ["guest", "host", "cleaner", "trust and safety agent"],
    "mascotas": ["pet owner", "vet", "clinic receptionist", "groomer"],
    "agricultura": ["farmer", "agronomist", "buyer", "field worker"],
    "gobierno": ["citizen", "civil servant", "records clerk", "department head"],
    "ong": ["donor", "campaign manager", "volunteer", "finance officer"],
    "musica": ["music student", "music teacher", "parent", "school coordinator"],
    "moda": ["customer", "stylist", "warehouse operator", "quality inspector"],
    "automotriz": ["car owner", "mechanic", "service adviser", "parts clerk"],
    "construccion": ["site manager", "worker", "client", "quantity surveyor"],
    "coworking": ["member", "community manager", "visitor", "facilities technician"],
    "legal": ["lawyer", "paralegal", "client", "billing clerk"],
    "cuidado": ["family member", "caregiver", "care coordinator", "nurse"],
    "reciclaje": ["household member", "collection driver", "plant operator", "route planner"],
    "gaming": ["player", "moderator", "tournament organiser", "clan leader"],
    "supermercado": ["shopper", "picker", "store manager", "delivery driver"],
    "farmacia": ["patient", "pharmacist", "courier", "pharmacy manager"],
    "belleza": ["client", "stylist", "salon owner", "receptionist"],
    "parqueo": ["driver", "parking inspector", "city operator", "resident"],
    "lavanderia": ["customer", "driver", "plant operator", "shift supervisor"],
    "cine": ["moviegoer", "cinema manager", "box office staff", "programmer"],
    "movilidad": ["rider", "field technician", "city operator", "fleet manager"],
    "colegio": ["parent", "teacher", "principal", "school secretary"],
    "banca": ["account holder", "support agent", "compliance officer", "branch manager"],
    "telecom": ["subscriber", "support agent", "retention specialist", "network engineer"],
}
faltan = set(SECTORES) - set(ROLES)
assert not faltan, f"sectores sin plantel de roles: {faltan}"

# --- plantillas: todas nombran 2 o 3 publicos ------------------------------
PLANTILLAS = [
    "{articulo} for {p1} and {p2}, with {f1} and {f2}.",
    "{articulo} where {p1} can {verbo} {f1}, and {p2} handle {f2}. It also needs {f3}.",
    "A platform used by {p1}, {p2} and {p3}, offering {f1}, {f2} and {f3}.",
    "{articulo} focused on {f1}, with {f2} for {p1} and {f3} for {p2}.",
    "{articulo} serving {p1} and {p2}. The main needs are {f1}, {f2} and {f3}.",
    "We are building {articulo_min} for {p1}, {p2} and {p3}. It must support {f1} and {f2}.",
    "{articulo} that lets {p1} {verbo} {f1}, while {p2} follow {f2}.",
    "A mobile app for {p1} and {p2}: {f1}, {f2} and {f3}.",
    "A web system where {p1} {verbo} {f1} and {p2} review {f2}, including {f3}.",
    "{articulo} with {f1} and {f2}. Both {p1} and {p2} use it daily; {p3} need {f3}.",
]
# Misma plantilla con el publico QUITADO, en el mismo orden. Las descripciones
# que llegan a Agix muchas veces no dicen para quien es el producto, y si todos
# los ejemplos de entrenamiento lo dicen, el modelo aprende a COPIAR un
# sustantivo de la entrada al hueco del rol en vez de elegir un actor plausible
# del dominio. Cada descripcion genera tambien su variante sin publico, con las
# MISMAS historias como objetivo, para ensenar esa invariancia.
PLANTILLAS_SIN = [
    "{articulo} with {f1} and {f2}.",
    "{articulo} that must handle {f1} and {f2}. It also needs {f3}.",
    "A platform offering {f1}, {f2} and {f3}.",
    "{articulo} focused on {f1}, with {f2} and {f3}.",
    "{articulo}. The main needs are {f1}, {f2} and {f3}.",
    "We are building {articulo_min}. It must support {f1} and {f2}.",
    "{articulo} that handles {f1} and follows {f2}.",
    "A mobile app with {f1}, {f2} and {f3}.",
    "A web system for {f1} and the review of {f2}, including {f3}.",
    "{articulo} with {f1} and {f2}. It also needs {f3}.",
]
assert len(PLANTILLAS_SIN) == len(PLANTILLAS)
VERBOS = ["manage", "handle", "set up", "keep track of", "organise"]
POR_SECTOR = 20


def plural(rol):
    if rol.endswith("staff"):
        return rol
    if rol.endswith("y") and rol[-2] not in "aeiou":
        return rol[:-1] + "ies"
    if rol.endswith(("s", "x", "ch", "sh")):
        return rol + "es"
    return rol + "s"


casos = []
for sector, (articulo, _pub, feats) in SECTORES.items():
    # pares (plantilla, combo) sin repeticion: si se recorren por separado con
    # `i % len`, la plantilla y el combo vuelven a coincidir a las 10 vueltas y
    # sale una descripcion identica, que el deduplicador luego borra.
    pares = [(t, c) for t, c in itertools.product(range(len(PLANTILLAS)),
                                                  itertools.combinations(feats, 3))]
    random.shuffle(pares)
    roles = ROLES[sector]
    for i in range(POR_SECTOR):
        plantilla, (f1, f2, f3) = pares[i % len(pares)]
        # 3 actores del plantel, rotando para que todos aparezcan por igual
        elegidos = [roles[(i + j) % len(roles)] for j in range(3)]
        p1, p2, p3 = (plural(r) for r in elegidos)
        art = articulo[0].upper() + articulo[1:]
        verbo = random.choice(VERBOS)      # uno solo, para no mover la semilla
        campos = dict(articulo=art, articulo_min=articulo, p1=p1, p2=p2, p3=p3,
                      verbo=verbo, f1=f1, f2=f2, f3=f3)
        txt = PLANTILLAS[plantilla].format(**campos)
        casos.append(dict(id=f"v3-{sector}-{i}", sector=sector, descripcion=txt,
                          rasgos=[f1, f2, f3],
                          descripcion_sin=PLANTILLAS_SIN[plantilla].format(**campos),
                          roles=roles))

# quitar casi duplicadas con el SBERT del propio proyecto
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(f"{BASE}/models/sbert_agix_v1",
                         device=os.environ.get("AGIX_DEVICE", "cuda"))
E = sb.encode([c["descripcion"] for c in casos], convert_to_tensor=True,
              normalize_embeddings=True, show_progress_bar=False)
S = (E @ E.T).float().cpu().numpy()
np.fill_diagonal(S, 0)
quedan, vistos = [], np.zeros(len(casos), bool)
for i in range(len(casos)):
    if vistos[i]:
        continue
    vistos |= (S[i] >= 0.93)
    quedan.append(casos[i])

json.dump(quedan, open(f"{BASE}/data/synth/descripciones_v3.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
json.dump(ROLES, open(f"{BASE}/data/synth/roles_v3.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)

c = collections.Counter(x["sector"] for x in quedan)
print(f"{len(casos)} generadas, {len(quedan)} tras quitar casi duplicadas, "
      f"{len(SECTORES)} sectores ({len(NUEVOS)} nuevos)")
print(f"por sector: min {min(c.values())} max {max(c.values())} "
      f"mediana {sorted(c.values())[len(c)//2]}")
print(f"similitud media entre descripciones distintas: {S[S > 0].mean():.3f}")
print(f"actores distintos en los planteles: {len({r for v in ROLES.values() for r in v})}")
for x in quedan[:3] + quedan[-3:]:
    print(f"  [{x['sector']}] {x['descripcion']}")
