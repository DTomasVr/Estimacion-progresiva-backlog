"""Side-by-side reading material: the old models against the distilled ones, on
descriptions chosen to answer three different questions.

  A. the actual use case   -- consumer products close to the synthetic sectors
  B. generalisation        -- consumer products in sectors NOT in the taxonomy,
                              to see whether it learned the register or just
                              memorised 30 sectors
  C. no regression         -- technical descriptions, the corpus's home ground

Prints, does not score: the point is for a human to read them.

  python src/45_leer_salidas.py [viejo|nuevo|ambos]
"""
import importlib.util
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AGIX_DEVICE", "cuda")
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

NUEVO_ABAJO = os.environ.get("AGIX_ABAJO",
    f"{BASE}/models/mt5-small-pruned_hist_agix_v1_mix_agix_v1")
NUEVO_ARRIBA = os.environ.get("AGIX_ARRIBA",
    f"{BASE}/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_mixa_agix_v1")

CASOS = [
    ("A · caso real de Agix", "An e-commerce platform for selling audiobooks and ebooks."),
    ("A · caso real de Agix", "A web application to book home cleaning services, with recurring "
                              "appointments and in-app payments."),
    ("B · sector NO entrenado", "An app to find and pay for street parking, with a timer, "
                                "reminders before the meter expires and monthly invoices."),
    ("B · sector NO entrenado", "A wedding planning tool where couples manage the guest list, "
                                "seating chart, vendor payments and RSVPs."),
    ("B · sector NO entrenado", "A language exchange app that matches learners by level and "
                                "interests, with scheduled video calls and progress badges."),
    ("C · dominio del corpus", "An authentication service with OAuth2, session management, token "
                               "refresh and an admin panel for users and permissions."),
]

cual = sys.argv[1] if len(sys.argv) > 1 else "ambos"
variantes = []
if cual in ("viejo", "ambos"):
    variantes.append(("VIEJO (solo datos reales)", pl.M_ARRIBA, pl.M_ABAJO))
if cual in ("nuevo", "ambos"):
    if not os.path.isdir(NUEVO_ABAJO):
        print(f"todavia no existe {NUEVO_ABAJO}; entrena primero")
    else:
        variantes.append(("NUEVO (real + destilado)", NUEVO_ARRIBA, NUEVO_ABAJO))

for etiqueta, arriba, abajo in variantes:
    pl.M_ARRIBA, pl.M_ABAJO = arriba, abajo
    P = pl.Pipeline(conjunto="no", traducir=False)      # un solo generador: lo que se despliega
    print(f"\n{'#'*110}\n##### {etiqueta}\n{'#'*110}")
    for grupo, desc in CASOS:
        r = P(desc)
        H = r["historias"]
        votos = [h["votos"] for h in H]
        rel = [h["relevancia"] for h in H]
        print(f"\n=== [{grupo}] {desc}")
        print(f"    {len(H)} historias, {r['n_tareas']} tareas, {r['tiempos']['total']:.1f}s | "
              f"votos max {max(votos)} media {sum(votos)/len(votos):.1f} | "
              f"relevancia media {sum(rel)/len(rel):.3f}")
        for h in H[:6]:
            print(f"  [{h['votos']}v {h['relevancia']:.2f}] {h['historia'][:115]}")
            for t in h["tareas"][:3]:
                print(f"       - {t[:100]}")
    del P
    import torch
    torch.cuda.empty_cache()
