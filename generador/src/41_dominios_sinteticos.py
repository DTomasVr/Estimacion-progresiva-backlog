"""Phase 1 of the distillation pilot: build varied PRODUCT descriptions, the kind
Agix actually receives, which is exactly what the Jira corpus lacks.

They are built from a seeded taxonomy instead of asked from the LLM, so the
coverage of sectors is controlled rather than whatever the model feels like
producing. Near-duplicates are then dropped with the project's own SBERT.

out: data/synth/descripciones.json
"""
import itertools
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/data/synth", exist_ok=True)
SEED = 20260922
random.seed(SEED)

# sector -> (what the product is, who uses it, things it must do)
SECTORES = {
    "e-commerce": ("an online store", ["shoppers", "sellers", "store managers"],
                   ["product catalogue with variants", "shopping cart and checkout", "discount coupons",
                    "order tracking", "product reviews and ratings", "wishlists", "refunds and returns",
                    "inventory alerts"]),
    "marketplace": ("a marketplace", ["buyers", "independent sellers", "moderators"],
                    ["seller onboarding", "commission settlement", "dispute resolution",
                     "search with filters", "seller reputation", "chat between buyer and seller"]),
    "libros y audio": ("a digital bookstore", ["readers", "publishers", "narrators"],
                       ["audiobook and ebook catalogue", "sample previews", "personal library",
                        "offline downloads", "reading progress sync", "royalty reports for publishers"]),
    "streaming": ("a video streaming service", ["viewers", "content editors"],
                  ["personalised recommendations", "watchlists", "parental controls",
                   "subtitles and audio tracks", "continue watching across devices"]),
    "delivery": ("a food delivery app", ["customers", "couriers", "restaurants"],
                 ["live order tracking", "courier assignment", "restaurant menus with schedules",
                  "tips and ratings", "scheduled deliveries"]),
    "turismo": ("a travel booking platform", ["travellers", "hotel managers"],
                ["room availability calendar", "cancellation policies", "multi-city itineraries",
                 "loyalty points", "traveller reviews"]),
    "salud": ("a clinic management system", ["patients", "doctors", "receptionists"],
              ["appointment scheduling", "electronic medical records", "prescriptions",
               "lab result delivery", "insurance claims", "reminders by SMS"]),
    "fitness": ("a fitness tracking app", ["members", "trainers"],
                ["workout plans", "progress charts", "class booking", "wearable sync",
                 "nutrition logging"]),
    "educacion": ("an online course platform", ["students", "instructors", "coordinators"],
                  ["enrolment and payment", "video lessons with progress", "quizzes and grading",
                   "certificates", "discussion forums", "assignment submission"]),
    "finanzas": ("a personal finance app", ["account holders", "support agents"],
                 ["bank account linking", "spending categories", "budget alerts",
                  "recurring payments", "shared expenses", "monthly reports"]),
    "seguros": ("an insurance portal", ["policyholders", "brokers", "claim adjusters"],
                ["quote simulator", "policy renewal", "claim filing with photos",
                 "document vault", "premium payment plans"]),
    "inmobiliaria": ("a real estate listing site", ["buyers", "agents", "landlords"],
                     ["property search with map", "visit scheduling", "mortgage simulator",
                      "listing photos and floor plans", "contract signing"]),
    "rrhh": ("a recruiting platform", ["candidates", "recruiters", "hiring managers"],
             ["job postings", "application tracking", "interview scheduling", "candidate scoring",
              "offer letters"]),
    "eventos": ("an event ticketing platform", ["attendees", "organisers"],
                ["seat selection", "QR check-in", "refund windows", "waiting lists",
                 "promoter dashboards"]),
    "logistica": ("a shipment tracking system", ["senders", "warehouse staff", "drivers"],
                  ["label printing", "route planning", "proof of delivery", "returns handling",
                   "warehouse stock counts"]),
    "restaurantes": ("a restaurant point of sale", ["waiters", "kitchen staff", "owners"],
                     ["table orders", "split bills", "kitchen display", "daily sales reports",
                      "stock deduction per dish"]),
    "inmuebles alquiler": ("a short-term rental platform", ["guests", "hosts"],
                           ["availability calendar", "instant booking", "cleaning fees",
                            "guest verification", "damage deposits"]),
    "mascotas": ("a veterinary clinic app", ["pet owners", "vets"],
                 ["pet profiles", "vaccination reminders", "appointment booking",
                  "medical history", "online pharmacy"]),
    "agricultura": ("a farm management tool", ["farmers", "buyers", "agronomists"],
                    ["crop planning", "harvest records", "direct sales to restaurants",
                     "weather alerts", "traceability per batch"]),
    "gobierno": ("a citizen services portal", ["citizens", "civil servants"],
                 ["appointment booking", "document requests", "payment of fees",
                  "status tracking of procedures", "digital signatures"]),
    "ong": ("a donation platform", ["donors", "campaign managers"],
            ["recurring donations", "campaign pages", "impact reports", "tax receipts",
             "volunteer sign-up"]),
    "musica": ("a music lessons marketplace", ["students", "teachers"],
               ["lesson scheduling", "video calls", "practice tracking", "teacher profiles",
                "package payments"]),
    "moda": ("a fashion rental service", ["customers", "stylists", "warehouse staff"],
             ["size profiles", "rental periods", "cleaning cycle tracking", "style recommendations",
              "return shipping labels"]),
    "automotriz": ("a car workshop management system", ["car owners", "mechanics", "service advisers"],
                   ["service booking", "repair estimates", "spare part orders",
                    "maintenance history", "pick-up and delivery"]),
    "construccion": ("a construction site tracking app", ["site managers", "workers", "clients"],
                     ["daily progress reports", "photo evidence", "material requests",
                      "safety checklists", "subcontractor invoices"]),
    "coworking": ("a coworking space booking app", ["members", "community managers"],
                  ["desk and room booking", "membership plans", "visitor passes",
                   "usage billing", "event calendar"]),
    "legal": ("a legal case management tool", ["lawyers", "paralegals", "clients"],
              ["case timelines", "document templates", "billable hours", "court date reminders",
               "client portal"]),
    "cuidado": ("a home care service platform", ["families", "caregivers", "coordinators"],
                ["caregiver matching", "shift scheduling", "visit check-in", "care notes",
                 "recurring appointments", "in-app payments"]),
    "reciclaje": ("a recycling pickup service", ["households", "collection drivers", "plant staff"],
                  ["pickup scheduling", "material weighing", "reward points",
                   "route optimisation", "monthly statements"]),
    "gaming": ("a game community platform", ["players", "moderators", "tournament organisers"],
               ["player profiles", "tournament brackets", "match reporting", "clan management",
                "in-game item trading"]),
}

PLANTILLAS = [
    "{articulo} for {publico}, with {f1} and {f2}.",
    "{articulo} where {publico} can {verbo} {f1}, plus {f2} and {f3}.",
    "A platform that offers {f1}, {f2} and {f3}, used by {publico}.",
    "{articulo} focused on {f1}, with {f2} for {publico}.",
    "{articulo}. The main needs are {f1}, {f2} and {f3}.",
    "We are building {articulo_min}. It must support {f1} and {f2}, mainly for {publico}.",
    "{articulo} that lets {publico} {verbo} {f1}. It also needs {f2}.",
    "A mobile app for {publico}: {f1}, {f2} and {f3}.",
    "A web system for {publico} to {verbo} {f1}, including {f2}.",
    "{articulo} with {f1}. {publico} also need {f2} and {f3}.",
    "An internal tool for {publico}, covering {f1} and {f2}.",
]
VERBOS = ["manage", "handle", "set up", "keep track of", "organise"]

casos = []
for sector, (articulo, publicos, feats) in SECTORES.items():
    combos = list(itertools.combinations(feats, 3))
    random.shuffle(combos)
    for i in range(22):
        f1, f2, f3 = combos[i % len(combos)]
        art = articulo[0].upper() + articulo[1:]
        txt = PLANTILLAS[i % len(PLANTILLAS)].format(
            articulo=art, articulo_min=articulo, publico=random.choice(publicos),
            verbo=random.choice(VERBOS), f1=f1, f2=f2, f3=f3)
        casos.append(dict(id=f"{sector}-{i}", sector=sector, descripcion=txt))

# drop near-duplicate descriptions with the project's own SBERT
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(f"{BASE}/models/sbert_agix_v1", device="cuda")
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

json.dump(quedan, open(f"{BASE}/data/synth/descripciones.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(f"{len(casos)} generadas, {len(quedan)} tras quitar casi duplicadas, "
      f"{len(SECTORES)} sectores")
print(f"similitud media entre descripciones distintas: {S[S > 0].mean():.3f}")
for c in quedan[:4]:
    print(f"  [{c['sector']}] {c['descripcion']}")
