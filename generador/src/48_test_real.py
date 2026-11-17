"""An INDEPENDENT test set of real, human-written user stories in consumer domains.

Why it matters: the distillation pilot was trained on backlogs written by qwen and
evaluated against references written by qwen, so part of the gain could be teacher
mimicry. This set is written by people, published by a different research group,
and none of it is in our training data.

Source: Bragilovski, van Can, Dalpiaz, Sturm (2024), "User Stories Dataset for
Domain Model Derivation", Zenodo 12740518, CC BY 4.0.

Of its 9 projects, 3 (camperplus, planningpoker, recycling) overlap 75-100 % with
the Dalpiaz set we already train on, so they are EXCLUDED. The remaining 6 are new.

The descriptions below are written by hand from reading each project's stories,
the same role the curator paragraph plays for Dalpiaz: the model needs an input,
and this is what a user would plausibly type. They were written WITHOUT looking at
any model output.

out: data/reales/test_real.json
"""
import json
import os
import re
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = f"{BASE}/data/reales/bragilovski.zip"
PREF = "Material-RE24-main/A Benchmark for Domain Model/User Stories/"
EXCLUIDOS = {"camperplus.txt", "planningpoker.txt", "recycling.txt"}   # solapan con Dalpiaz

DESCRIPCIONES = {
    "Fish_Chips.txt": (
        "A fish and chips restaurant order system where the kitchen sees incoming orders and their "
        "status, staff handle special requests, home delivery drivers get the address on the ticket, "
        "and the owner sees prices and daily revenue."),
    "Ticket.txt": (
        "An event ticketing platform where visitors buy tickets and see their upcoming events, "
        "artists manage their event pages and descriptions, and the company validates tickets. "
        "Guests can buy without registering."),
    "grocery.txt": (
        "A staff scheduling and hiring system for a grocery chain, where managers schedule shifts and "
        "review employee contracts, applicants register and complete their hiring, employees check "
        "their schedule, and office staff handle complaints."),
    "school.txt": (
        "A school management system with accounts for students, teachers, guardians and "
        "administrators, covering timetables and change notifications, messaging with mentors, "
        "uploading course files, and reporting a pupil's absence."),
    "sports.txt": (
        "A gym membership system where customers open an account, choose and extend a subscription "
        "and pay, trainers schedule lessons in available rooms and can cancel them, and admins "
        "manage trainers and users."),
    "supermarket.txt": (
        "A supermarket online shopping app where customers search products, see when a sold-out item "
        "returns, repeat a previous order, subscribe to newsletters and push notifications about "
        "discounts, and contact customer service; employees publish recipes and confirm delivery "
        "times."),
}

z = zipfile.ZipFile(ZIP)
salida = []
for archivo, desc in DESCRIPCIONES.items():
    assert archivo not in EXCLUIDOS
    texto = z.read(PREF + archivo).decode("utf-8", errors="replace")
    historias = []
    for linea in texto.split("\n"):
        linea = re.sub(r"\s+", " ", linea.replace("�", "'")).strip()
        if len(linea) > 25 and re.match(r"^as an?\s", linea, re.I):
            historias.append(linea)
    salida.append(dict(proyecto=archivo.replace(".txt", "").lower(), descripcion=desc,
                       historias=historias, fuente="Bragilovski2024 CC BY 4.0"))
    print(f"  {archivo:<18} {len(historias):3d} historias")

json.dump(salida, open(f"{BASE}/data/reales/test_real.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(f"\n{len(salida)} proyectos, {sum(len(x['historias']) for x in salida)} historias humanas reales")
print(f"escrito data/reales/test_real.json")
