"""Genera y ENSENA la salida del pipeline en 7 dominios de prueba fijos.

Es la herramienta principal para juzgar un cambio: en este proyecto varias mejoras
aparentes en las metricas resultaron ser el backlog empobreciendose, asi que se lee el
texto ANTES de mirar los numeros.

  python src/92_ver_salida.py <etiqueta>            los 7 dominios
  python src/92_ver_salida.py <etiqueta> banca bodas    solo esos

Compara dos corridas:
  python src/92_ver_salida.py --comparar antes despues

out: results/92_<etiqueta>.txt  y  results/92_<etiqueta>.json
"""
import collections
import importlib.util
import io
import itertools
import json
import os
import re
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PBE_DEVICE", "cuda")
os.environ.setdefault("PBE_SEED", "20260926")      # sin esto no se puede comparar
os.makedirs(f"{BASE}/results", exist_ok=True)

DOMINIOS = {
    "banca": "Una app de banca digital donde los clientes abren una cuenta con verificacion "
             "de identidad, programan transferencias, congelan su tarjeta y reciben alertas "
             "de fraude.",
    "bodas": "Una plataforma para organizar bodas donde los novios contratan proveedores, "
             "arman la lista de invitados con sus restricciones de comida y controlan el "
             "presupuesto.",
    "farmacia": "Una app de farmacia donde los pacientes suben su receta, buscan medicamentos "
                "con equivalentes y piden entrega a domicilio, y el farmaceutico valida los "
                "controlados.",
    "limpieza": "Una aplicacion web para reservar servicios de limpieza a domicilio, con citas "
                "recurrentes, calificacion de los trabajadores y pagos en la app.",
    "tecnico": "Un servicio de integracion continua que ejecuta pipelines de compilacion en "
               "agentes, guarda los artefactos generados y avisa al equipo cuando una rama falla.",
    "colegio": "Un sistema de gestion escolar con cuentas para estudiantes, profesores, "
               "apoderados y administracion, con horarios, mensajeria con el tutor y reporte "
               "de inasistencias.",
    "restaurante": "Un sistema de pedidos para un restaurante, donde la cocina ve los pedidos y "
                   "su estado, el personal atiende peticiones especiales y el dueno consulta "
                   "los ingresos del dia.",
}
RE_ROL = re.compile(r"^\s*as an?\s+([^,]+?)\s*,", re.I)
PAT_TRIPLE = re.compile(r"^\s*as an?\s+[^,]+,\s*i want\s+.+,\s*so that\s+.+", re.I)


def cargar_pipeline():
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    for var, attr in (("PBE_ARRIBA", "M_ARRIBA"), ("PBE_ABAJO", "M_ABAJO")):
        if os.environ.get(var):
            setattr(pl, attr, os.environ[var])
    return pl


def medir(pl, sb, x):
    """las cifras que este proyecto usa para juzgar un backlog"""
    H = x["historias_en"]
    n = len(H)
    E = sb.encode(H, normalize_embeddings=True, show_progress_bar=False)
    S = E @ E.T
    marc = [False] * n
    for i, j in itertools.combinations(range(n), 2):
        if S[i, j] > 0.85:
            marc[i] = marc[j] = True
    roles = collections.Counter()
    for h in H:
        m = RE_ROL.match(h)
        roles[m.group(1).lower() if m else "?"] += 1
    tds = [t for ts in x["tareas_en"] for t in ts]
    idx = [i for i, ts in enumerate(x["tareas_en"]) for _ in ts]
    atr = 0.0
    if tds:
        Et = sb.encode(tds, normalize_embeddings=True, show_progress_bar=False)
        M = Et @ E.T
        atr = 100 * sum(1 for j, i in enumerate(idx) if int(np.argmax(M[j])) == i) / len(tds)
    eco = sum(1 for ts in x["tareas"] for t in ts if pl.eco_por_raiz(t))
    return dict(historias=n, roles=len(roles),
                dominante=100 * roles.most_common(1)[0][1] / n,
                casi_dup=100 * sum(marc) / n, tareas=len(tds), atribucion=atr,
                eco=eco, formato=100 * sum(1 for h in H if PAT_TRIPLE.match(h)) / n,
                reparto=" > ".join(f"{k} x{v}" for k, v in roles.most_common()))


def generar(etiqueta, nombres):
    pl = cargar_pipeline()
    P = pl.Pipeline(traducir=True)
    print(f"arriba: {os.path.basename(pl.M_ARRIBA)}")
    print(f"abajo : {os.path.basename(pl.M_ABAJO)}")
    print(f"fichas: rol={'si' if P.planificador is not None else 'NO'}  "
          f"funcion={'si' if P.usa_funcion else 'NO'}"
          + ("    <-- OJO: se activan por el NOMBRE del modelo"
             if not (P.planificador is not None and P.usa_funcion) else ""))
    print(f"semilla: {os.environ['PBE_SEED']}\n")
    lineas, datos, tabla = [], {}, []
    for nom in nombres:
        d = DOMINIOS[nom]
        r = P(d, k_historias=12, n_muestras=36)
        H = r["historias"]
        x = dict(descripcion=d,
                 historias=[h.get("historia_es") or h["historia"] for h in H],
                 historias_en=[h["historia"] for h in H],
                 tareas=[(h.get("tareas_es") or h["tareas"]) for h in H],
                 tareas_en=[h["tareas"] for h in H])
        datos[nom] = x
        m = medir(pl, P.sb, x)
        tabla.append((nom, m))
        cab = (f"[{nom}] {m['historias']} historias, {m['tareas']} tareas, "
               f"{m['roles']} roles, atribucion {m['atribucion']:.0f} %, "
               f"{r['tiempos']['total']:.1f}s")
        bloque = ["=" * 96, cab, f"  roles: {m['reparto']}", d]
        for i, (h, ts) in enumerate(zip(x["historias"], x["tareas"]), 1):
            bloque.append(f"\n {i:>2}. {h}")
            bloque += [f"       - {t}" for t in ts]
        bloque.append("")
        print("\n".join(bloque))
        lineas += bloque
    print("=" * 96)
    print(f"{'dominio':<12} {'roles':>5} {'domin':>7} {'casi-dup':>9} {'tareas':>7} "
          f"{'atrib':>6} {'eco':>4} {'formato':>8}")
    for nom, m in tabla:
        print(f"{nom:<12} {m['roles']:>5} {m['dominante']:>6.0f}% {m['casi_dup']:>8.1f}% "
              f"{m['tareas']:>7} {m['atribucion']:>5.0f}% {m['eco']:>4} {m['formato']:>7.0f}%")
    prom = {k: np.mean([m[k] for _, m in tabla])
            for k in ("roles", "dominante", "casi_dup", "tareas", "atribucion", "formato")}
    resumen = (f"{'MEDIA':<12} {prom['roles']:>5.1f} {prom['dominante']:>6.0f}% "
               f"{prom['casi_dup']:>8.1f}% {prom['tareas']:>7.1f} "
               f"{prom['atribucion']:>5.0f}% {'':>4} {prom['formato']:>7.0f}%")
    print(resumen)
    print("\nreferencia humana: casi-duplicados 4,8 %   ·   formato: 100 % por construccion")
    lineas.append(resumen)
    io.open(f"{BASE}/results/92_{etiqueta}.txt", "w", encoding="utf-8").write("\n".join(lineas))
    json.dump(datos, io.open(f"{BASE}/results/92_{etiqueta}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n-> results/92_{etiqueta}.txt  y  .json")


def comparar(a, b):
    """que historias cambiaron entre dos corridas; con la misma semilla, lo que no
    cambia es lo que el cambio no toca"""
    A = json.load(io.open(f"{BASE}/results/92_{a}.json", encoding="utf-8"))
    B = json.load(io.open(f"{BASE}/results/92_{b}.json", encoding="utf-8"))
    for nom in A:
        if nom not in B:
            continue
        dif = [(x, y) for x, y in zip(A[nom]["historias_en"], B[nom]["historias_en"]) if x != y]
        print(f"{nom:<12} {len(dif):>2}/{len(A[nom]['historias_en'])} historias distintas")
        for x, y in dif[:2]:
            print(f"     -- {x[:92]}")
            print(f"     ++ {y[:92]}")


if __name__ == "__main__":
    if "--comparar" in sys.argv:
        i = sys.argv.index("--comparar")
        comparar(sys.argv[i + 1], sys.argv[i + 2])
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        etq = args[0] if args else "x"
        noms = [a for a in args[1:] if a in DOMINIOS] or list(DOMINIOS)
        generar(etq, noms)
