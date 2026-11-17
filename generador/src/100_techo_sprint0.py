"""E266 del estimador, con SUS funciones (corpus, simula), sobre los 58 backlogs validos de
calibracion: solo historias frente a historias + Sprint 0. Se corre desde Agix-Dataset."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
s = importlib.util.spec_from_file_location("e266", "C:/Users/User/Documents/Programacao/Agix-Dataset/src/266_techo_backlog_generado.py")
e266 = importlib.util.module_from_spec(s); s.loader.exec_module(e266)
s = importlib.util.spec_from_file_location("e6", "C:/Users/User/Documents/Programacao/Agix-Dataset/src/171_estimador_v6.py")
e6 = importlib.util.module_from_spec(s); s.loader.exec_module(e6)
est = e6.EstimadorV6(paquete=Path("models/estimador_agix_v7.joblib"), dispositivo="cpu")
J = json.load(open(r"F:\Software Task generation Model\data\calibracion\salidas_v4.json", encoding="utf-8"))
u, escala, _ = e266.corpus()
FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
print("factor de escala del Sprint 0:", FACTOR)
rng = np.random.default_rng(266)
filas = []
for k, v in J.items():
    if v["grupo"] not in ("taxonomia", "fuera"): continue
    th = [t for h in v["historias"] for t in h["tareas_en"]]
    t0 = [t for h in v["sprint0"] for t in h["tareas_en"]]
    mh = np.asarray(est._media_tarea(th), float)
    m0 = (np.asarray(est._media_tarea(t0), float) * FACTOR) if t0 else np.array([])
    a = e266.simula(mh, u, escala, rng)
    b = e266.simula(np.concatenate([mh, m0]), u, escala, rng)
    filas.append((k, len(th), len(t0), mh.sum(), m0.sum(), a[0], b[0], a[1], b[1]))
F = np.array([f[1:] for f in filas], float)
print(f"{len(filas)} backlogs · tareas de historias {F[:,0].mean():.1f} · del Sprint 0 {F[:,1].mean():.1f}")
print(f"horas (media del modelo): historias {F[:,2].mean():.1f} h · Sprint 0 {F[:,3].mean():.1f} h "
      f"({100*F[:,3].sum()/F[:,2].sum():.1f} % mas)")
print(f"\n{'':32}{'solo historias':>16}{'+ Sprint 0':>12}{'cambio':>9}")
print(f"{'techo (escala exacta)':<32}{100*F[:,4].mean():>15.1f}%{100*F[:,5].mean():>11.1f}%{100*(F[:,5].mean()-F[:,4].mean()):>+8.1f}")
print(f"{'en frio simulado':<32}{100*F[:,6].mean():>15.1f}%{100*F[:,7].mean():>11.1f}%{100*(F[:,7].mean()-F[:,6].mean()):>+8.1f}")
d = F[:,5] - F[:,4]
bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
print(f"\ntecho, diferencia emparejada: {100*d.mean():+.1f} puntos  IC95 [{100*np.quantile(bs,.025):+.1f}, {100*np.quantile(bs,.975):+.1f}]"
      f" · sube en {(d > 0).sum()} de {len(d)}")
