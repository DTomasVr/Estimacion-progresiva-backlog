"""Local demo server for the PBE generation pipeline. No extra dependencies:
the standard library serves the page, the pipeline runs in this same process.

    python src/33_servidor.py            # CPU (what the presentation laptop has)
    python src/33_servidor.py --gpu      # GPU if there is one
    python src/33_servidor.py --puerto 8123
    python src/33_servidor.py --sin-estimador      # solo el backlog, sin horas
    python src/33_servidor.py --total-solo         # horas del proyecto, 4,4 GB menos

Precomputed examples (results/33_ejemplos.json) answer instantly, so a demo
never depends on waiting or on having a GPU; typing a new description runs the
real pipeline and reports the time each stage took.

El estimador de esfuerzo (estimador_pbe_v6, el otro repositorio) se engancha por
src/34_estimacion.py y le pone horas a cada tarea, un subtotal a cada historia y un
total con intervalo al proyecto. Si no esta disponible, la pagina sigue funcionando
igual y lo dice; el backlog nunca depende de que haya estimador.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

BASE = os.environ.get("PBE_BASE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# la raiz del proyecto: la carpeta que contiene src/, models/, data/ y results/.
# Se deduce de la ubicacion de este archivo; PBE_BASE la sobreescribe si hace falta.
PUERTO = 8123
if "--puerto" in sys.argv:
    PUERTO = int(sys.argv[sys.argv.index("--puerto") + 1])
os.environ.setdefault("PBE_DEVICE", "cuda" if "--gpu" in sys.argv else "cpu")
if "--ligero" in sys.argv:
    os.environ["PBE_BEAMS"] = "1"

import importlib.util  # noqa: E402

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

EJEMPLOS = []
ruta_ej = f"{BASE}/results/33_ejemplos.json"
if os.path.exists(ruta_ej):
    EJEMPLOS = json.load(open(ruta_ej, encoding="utf-8"))

print(f"cargando el pipeline en {pl.DISPOSITIVO}...", flush=True)
if pl.DISPOSITIVO == "cpu":
    torch.set_num_threads(min(6, os.cpu_count() or 4))
CONJ = "no" if "--ligero" in sys.argv else pl.CONJUNTO
P = pl.Pipeline(conjunto=CONJ, traducir=True)
CANDADO = threading.Lock()

# El estimador de esfuerzo. Se carga DESPUES del generador y nunca lo tumba: si algo
# falla, la demo sigue dando backlogs y la pagina muestra el motivo.
EST = None
if "--sin-estimador" not in sys.argv:
    spec_e = importlib.util.spec_from_file_location("es34", f"{BASE}/src/34_estimacion.py")
    _e = importlib.util.module_from_spec(spec_e)
    try:
        spec_e.loader.exec_module(_e)
        if not _e.hay_codificador_base():
            print("  aviso: el codificador base no esta en el cache de Hugging Face; "
                  "la primera estimacion intentara descargarlo", flush=True)
        # El estimador puede ir en un dispositivo DISTINTO al del generador, y casi
        # siempre conviene: en CPU la cabeza por tarea son tres pasadas de e5-large
        # y tarda ~6 s para 40 tareas; en GPU son ~0,5 s. Los dos a la vez en GPU no
        # caben en 8 GB (3 x 2,24 GB del estimador + los del generador), asi que lo
        # normal es generador en CPU y estimador en GPU.
        dev_est = os.environ.get("PBE_EST_DEVICE") or (
            "cuda" if "--gpu-estimador" in sys.argv else pl.DISPOSITIVO)
        EST = _e.Estimacion(por_tarea="--total-solo" not in sys.argv,
                            dispositivo=dev_est)
        print(f"estimador: {'listo' if EST.disponible else 'NO disponible'} en "
              f"{EST.carga_s}s"
              + (f" ({EST.motivo})" if not EST.disponible else
                 ("" if EST.por_tarea else " · solo totales")), flush=True)
        if EST.disponible:
            # Calentar los codificadores AHORA y no en la primera peticion. Son de
            # carga perezosa: medido, la primera estimacion tardaba 31 s en CPU y la
            # mayor parte era abrir los modelos, no estimar.
            t_c = time.time()
            EST.estimar(dict(historias=[dict(tareas=["Add OAuth2 login with Google"])]))
            print(f"  codificadores calientes en {time.time() - t_c:.1f}s", flush=True)
            # Y estimar YA los ejemplos precalculados. Son el camino que se usa al
            # presentar: tienen que salir al instante, no despues de unos segundos de
            # "estimando horas". Se hace una sola vez al arrancar.
            t_c = time.time()
            n_ej = 0
            for ej in EJEMPLOS:
                r_ej = ej.get("resultado") or {}
                if r_ej.get("historias") and not r_ej.get("estimacion"):
                    EST.estimar(r_ej)
                    n_ej += 1
            if n_ej:
                print(f"  {n_ej} ejemplos estimados en {time.time() - t_c:.1f}s",
                      flush=True)
    except Exception as e:                           # noqa: BLE001
        print(f"estimador: NO disponible ({type(e).__name__}: {e})", flush=True)
        EST = None

print(f"listo en {P.carga_s}s -> http://127.0.0.1:{PUERTO}", flush=True)

if "--abrir" in sys.argv:
    # se abre AQUI, no en el lanzador: si el navegador arranca antes de que los
    # modelos esten cargados, lo primero que ve el usuario es un error de conexion
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{PUERTO}")

PAGINA = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PBE · generador de backlog</title>
<style>
:root{--fondo:#faf9f7;--papel:#fff;--tinta:#1d1c1a;--suave:#6b6762;--linea:#e6e2dc;
--acento:#2f6f5e;--acento2:#8a5a2b;--radio:10px}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--fondo:#17181a;
--papel:#1f2124;--tinta:#ecebe8;--suave:#a4a09a;--linea:#31343a;--acento:#69b39c;--acento2:#d3a06a}}
*{box-sizing:border-box}
body{margin:0;background:var(--fondo);color:var(--tinta);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.envoltura{max-width:960px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:1.5rem;margin:0 0 4px}
.sub{color:var(--suave);margin:0 0 24px;font-size:.92rem}
textarea{width:100%;min-height:92px;padding:12px;border:1px solid var(--linea);border-radius:var(--radio);
background:var(--papel);color:var(--tinta);font:inherit;resize:vertical}
.fila{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0 0}
button{border:0;border-radius:var(--radio);padding:10px 18px;font:inherit;font-weight:600;cursor:pointer;
background:var(--acento);color:#fff}
button.sec{background:transparent;color:var(--acento);border:1px solid var(--linea);font-weight:500}
button:disabled{opacity:.5;cursor:default}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}
.chip{border:1px solid var(--linea);background:var(--papel);border-radius:999px;padding:6px 12px;
font-size:.82rem;cursor:pointer;color:var(--suave)}
.chip:hover{border-color:var(--acento);color:var(--acento)}
.aviso{background:var(--papel);border:1px solid var(--linea);border-left:3px solid var(--acento2);
border-radius:var(--radio);padding:10px 14px;margin:18px 0;font-size:.88rem;color:var(--suave)}
.hist{background:var(--papel);border:1px solid var(--linea);border-radius:var(--radio);padding:14px 16px;margin:12px 0}
.hist h3{margin:0 0 8px;font-size:1rem;font-weight:600;line-height:1.4}
.hist h3 .sub-h{float:right;margin-left:14px;font-weight:600;font-size:.84rem;
color:var(--acento);font-variant-numeric:tabular-nums;white-space:nowrap}
.meta{font-size:.76rem;color:var(--suave);margin:0 0 8px}
ul{margin:8px 0 0;padding-left:20px}li{margin:3px 0}
li .h{display:inline-block;min-width:3.8em;margin-right:10px;text-align:right;
font-variant-numeric:tabular-nums;color:var(--acento2);font-size:.85rem}
.total{background:var(--papel);border:1px solid var(--linea);border-left:3px solid var(--acento);
border-radius:var(--radio);padding:14px 18px;margin:18px 0 6px;display:flex;
gap:24px;align-items:baseline;flex-wrap:wrap}
.total .cifra{font-size:1.9rem;font-weight:700;color:var(--acento);
font-variant-numeric:tabular-nums;line-height:1}
.total .det{font-size:.88rem;color:var(--suave)}
.total .det b{color:var(--tinta);font-variant-numeric:tabular-nums}
.total .nota{font-size:.76rem;color:var(--suave);line-height:1.5;flex-basis:100%;margin:2px 0 0}
.sinest{background:var(--papel);border:1px solid var(--linea);border-left:3px solid var(--acento2);
border-radius:var(--radio);padding:10px 14px;margin:18px 0 6px;font-size:.84rem;color:var(--suave)}
.tiempos{display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:var(--suave);
border-top:1px solid var(--linea);padding-top:10px;margin-top:20px}
.tiempos b{color:var(--tinta);font-variant-numeric:tabular-nums}
.cargando{color:var(--suave);font-size:.9rem}
.fuente{font-size:.76rem;color:var(--suave);margin-top:6px}
</style></head><body><div class="envoltura">
<h1>Generador de backlog</h1>
<p class="sub">Descripción del proyecto en español o inglés &rarr; historias de usuario, cada una con sus tareas
<b>y las horas que se estiman para cada una</b>. Dos modelos propios encadenados, sin llamadas a
servicios externos.</p>

<textarea id="desc" placeholder="Ej.: Una aplicación web para reservar servicios de limpieza a domicilio, con citas recurrentes y pagos en la app."></textarea>
<div class="fila">
  <button id="generar">Generar backlog</button>
  <button class="sec" id="es">Ver en español</button>
  <label style="font-size:.85rem;color:var(--suave)">historias:
    <select id="k" style="font:inherit;padding:4px 6px;border-radius:8px;border:1px solid var(--linea);background:var(--fondo);color:var(--tinta)">
      <option>12</option><option selected>20</option><option>30</option><option>40</option>
    </select></label>
  <span class="cargando" id="estado"></span>
</div>
<div class="chips" id="chips"></div>
<div class="aviso" id="aviso" style="display:none"></div>
<div id="salida"></div>
<div class="tiempos" id="tiempos" style="display:none"></div>
</div>
<script>
let ultimo=null, enEspanol=true;
const $=id=>document.getElementById(id);

fetch('/ejemplos').then(r=>r.json()).then(e=>{
  $('chips').innerHTML = e.map((x,i)=>`<span class="chip" data-i="${i}">${x.titulo}</span>`).join('');
  $('chips').onclick = ev => {
    const i = ev.target.dataset.i; if(i===undefined) return;
    $('desc').value = e[i].descripcion;
    const r = e[i].resultado;
    mostrar(r, true);
    // Los ejemplos vienen precalculados sin horas: se muestran al instante y la
    // estimacion llega despues, para no retrasar la demo.
    if(!r.estimacion) estimarAhora(r);
  };
});

async function estimarAhora(r){
  $('estado').textContent = 'Estimando horas…';
  try{
    const resp = await fetch('/estimar', {method:'POST', headers:{'Content-Type':'application/json'},
                                          body: JSON.stringify({historias:r.historias})});
    const j = await resp.json();
    r.estimacion = j.estimacion;            // se queda cacheada en el ejemplo
    if(j.historias) r.historias = j.historias;
    if(ultimo === r) pintar(r);
    $('estado').textContent = j.estimacion && j.estimacion.disponible ? '' : 'Sin estimación.';
  }catch(err){ $('estado').textContent = 'No se pudo estimar: ' + err.message; }
}

$('generar').onclick = async () => {
  const d = $('desc').value.trim();
  if(d.length < 15){ $('estado').textContent='Escribe una descripción algo más larga.'; return; }
  $('generar').disabled = true; $('estado').textContent = 'Generando (el modelo corre aquí mismo)…';
  const t0 = performance.now();
  try{
    const r = await fetch('/generar', {method:'POST', headers:{'Content-Type':'application/json'},
                                      body: JSON.stringify({descripcion:d, k:parseInt($('k').value)})});
    const j = await r.json();
    if(j.error){ $('estado').textContent = 'Error: ' + j.error; }
    else { mostrar(j, false); $('estado').textContent =
           `Listo en ${((performance.now()-t0)/1000).toFixed(1)} s`; }
  }catch(e){ $('estado').textContent = 'Error: ' + e.message; }
  $('generar').disabled = false;
};

$('es').onclick = () => { enEspanol = !enEspanol;
  $('es').textContent = enEspanol ? 'Ver en inglés' : 'Ver en español';
  if(ultimo) pintar(ultimo); };

function mostrar(j, precalc){
  ultimo = j;
  $('aviso').style.display = precalc ? 'block' : 'none';
  if(precalc) $('aviso').textContent = 'Ejemplo precalculado: se muestra al instante, sin generar.';
  pintar(j);
}

const hr = v => (v>=10 ? v.toFixed(0) : v.toFixed(1)).replace('.', ',') + ' h';

function panelTotal(e){
  if(!e) return '';
  if(!e.disponible) return `<div class="sinest">Sin estimación de horas: ${esc(e.motivo||'no disponible')}.
    El backlog de arriba no depende de ella.</div>`;
  const porTarea = e.por_tarea
    ? `Las horas de cada tarea son la <b>mediana</b>: el tiempo de una tarea típica.
       Los subtotales y el total son la <b>media</b>, que es lo único que se puede sumar,
       así que no cuadran con la suma de las tareas y eso es correcto.`
    : `Modo solo totales: no se cargaron los codificadores de la estimación por tarea.`;
  const est = e.factor_estilo && e.factor_estilo < 0.99
    ? ` Corregido por el estilo del generador (×${String(e.factor_estilo).replace('.',',')}):
        sin eso el total saldría un ${Math.round((1/e.factor_estilo-1)*100)} % alto.` : '';
  return `<div class="total">
    <div class="cifra">${hr(e.total_horas)}</div>
    <div class="det"><b>${String(e.dias_persona).replace('.',',')}</b> días-persona ·
      banda al ${Math.round(e.nivel*100)} %: <b>${hr(e.banda_baja)}</b> a <b>${hr(e.banda_alta)}</b>
      · ${e.tareas} tareas${e.cerradas?` · ${e.cerradas} ya cerradas`:''}</div>
    <p class="nota">${porTarea}${est} Estimación en ${String(e.segundos).replace('.',',')} s.</p>
  </div>`;
}

function pintar(j){
  const hs = j.historias || [];
  const est = j.estimacion;
  const hay = est && est.disponible;
  $('salida').innerHTML = panelTotal(est) + (hs.map(h => {
    const texto = enEspanol && h.historia_es ? h.historia_es : h.historia;
    const tareas = enEspanol && h.tareas_es ? h.tareas_es : (h.tareas||[]);
    const org = (h.origen||[]).map(o=>o.proyecto).join(', ');
    const sub = hay && h.horas_suma!==undefined
      ? `<span class="sub-h">${hr(h.horas_suma)}</span>` : '';
    const ht = h.horas_tareas || [];
    return `<div class="hist"><h3>${sub}${esc(texto)}</h3>
      <p class="meta">${h.votos||1} de ${j.n_banco||'?'} muestreos coincidieron${h.relevancia?` · relevancia ${h.relevancia}`:''}</p>
      <ul>${tareas.map((t,i)=>`<li>${ht[i]!==undefined?`<span class="h">${hr(ht[i])}</span>`:''}${esc(t)}</li>`).join('')}</ul>
      ${org?`<p class="fuente">Condicionada por proyectos reales: ${esc(org)}</p>`:''}</div>`;
  }).join('') || '<p class="sub">Sin resultados.</p>');

  const t = j.tiempos || {};
  const orden = [['recuperar_arriba','recuperar'],['generar_historias','historias'],
                 ['deduplicar_historias','dedup. historias'],['recuperar_abajo','recuperar tareas'],
                 ['generar_tareas','tareas'],['deduplicar_tareas','dedup. tareas'],
                 ['traducir','traducir'],['total','TOTAL']];
  $('tiempos').style.display = 'flex';
  $('tiempos').innerHTML = orden.filter(([k])=>t[k]!==undefined)
    .map(([k,n])=>`<span>${n}: <b>${t[k].toFixed(2)} s</b></span>`).join('')
    + (hay?`<span>estimar: <b>${est.segundos.toFixed(2)} s</b></span>`:'')
    + `<span>historias: <b>${hs.length}</b></span><span>tareas: <b>${j.n_tareas||0}</b></span>`;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
</script></body></html>"""


import re  # noqa: E402

UMBRAL_INTRA = 0.95


def degenerada(t):
    """delega en el pipeline: una sola implementacion del filtro"""
    return pl.degenerada_texto(t)


def limpiar_resultado(r):
    """Same two filters used for the precomputed examples, so live generation and
    presets look alike: drop degenerate items, and tasks that repeat another task
    of the SAME story (what 19_pipeline only does across stories -- annex A14)."""
    import numpy as np
    H = [h for h in r.get("historias", [])
         if not degenerada(h["historia"]) and not degenerada(h.get("historia_es") or h["historia"])]
    for h in H:
        ts = h.get("tareas") or []
        if not ts:
            continue
        if len(ts) > 1:
            E = P.sb.encode(ts, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=False)
            S = (E @ E.T).float().cpu().numpy()
            np.fill_diagonal(S, 0)
            quedan = [i for i in range(len(ts)) if not (S[i, :i] >= UMBRAL_INTRA).any()]
        else:
            quedan = [0]
        quedan = [i for i in quedan if not degenerada(ts[i])]
        h["tareas"] = [ts[i] for i in quedan]
        if h.get("tareas_es"):
            h["tareas_es"] = [h["tareas_es"][i] for i in quedan if i < len(h["tareas_es"])]
    r["historias"] = [h for h in H if h["tareas"]]
    r["n_tareas"] = sum(len(h["tareas"]) for h in r["historias"])
    return r


class Manejador(BaseHTTPRequestHandler):
    def _enviar(self, codigo, cuerpo, tipo="application/json; charset=utf-8"):
        datos = cuerpo if isinstance(cuerpo, bytes) else cuerpo.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        ruta = urlparse(self.path).path
        if ruta == "/":
            self._enviar(200, PAGINA, "text/html; charset=utf-8")
        elif ruta == "/ejemplos":
            self._enviar(200, json.dumps(EJEMPLOS, ensure_ascii=False))
        elif ruta == "/salud":
            self._enviar(200, json.dumps(dict(
                dispositivo=pl.DISPOSITIVO, conjunto=CONJ, carga_s=P.carga_s,
                ejemplos=len(EJEMPLOS),
                estimador=dict(disponible=bool(EST and EST.disponible),
                               por_tarea=bool(EST and EST.por_tarea),
                               carga_s=EST.carga_s if EST else None,
                               motivo=EST.motivo if EST else "arrancado sin estimador"))))
        else:
            self._enviar(404, json.dumps(dict(error="no existe")))

    @staticmethod
    def _anclaje(pet):
        """cerradas, respuestas y previo_equipo de la peticion (handoff del estimador,
        13/11/2026): las listas van en el orden de las tareas recorriendo las
        historias, con null donde no haya dato."""
        return {k: pet[k] for k in ("cerradas", "respuestas", "previo_equipo") if
                pet.get(k) is not None}

    def _desplazamiento(self, pet):
        """Lo que la web guarda de un proyecto TERMINADO para el previo del siguiente.
        Recibe el backlog tal como se genero y las horas reales de sus tareas."""
        if EST is None:
            return self._enviar(503, json.dumps(dict(error="servidor sin estimador")))
        hs, horas = pet.get("historias") or [], pet.get("horas") or []
        if not hs or not horas:
            return self._enviar(400, json.dumps(dict(error="faltan historias u horas")))
        with CANDADO:
            d = EST.desplazamiento(dict(historias=hs), horas)
        return self._enviar(200, json.dumps(dict(desplazamiento=round(d, 4))))

    def _estimar(self, pet):
        """Pone horas a un backlog YA generado. Lo usan los ejemplos precalculados,
        que se pintan al instante y reciben la estimacion despues."""
        if EST is None:
            return self._enviar(200, json.dumps(
                dict(estimacion=dict(disponible=False, motivo="servidor sin estimador"))))
        hs = pet.get("historias") or []
        if not hs:
            return self._enviar(400, json.dumps(dict(error="sin historias")))
        with CANDADO:
            r = EST.estimar(dict(historias=hs), **self._anclaje(pet))
        return self._enviar(200, json.dumps(r, ensure_ascii=False))

    def do_POST(self):
        ruta = urlparse(self.path).path
        if ruta not in ("/generar", "/estimar", "/desplazamiento"):
            return self._enviar(404, json.dumps(dict(error="no existe")))
        n = int(self.headers.get("Content-Length", 0))
        try:
            pet = json.loads(self.rfile.read(n) or b"{}")
            if ruta == "/estimar":
                return self._estimar(pet)
            if ruta == "/desplazamiento":
                return self._desplazamiento(pet)
            d = (pet.get("descripcion") or "").strip()
            if len(d) < 15:
                return self._enviar(400, json.dumps(dict(error="descripcion demasiado corta")))
            t = time.time()
            with CANDADO:                       # one generation at a time: a single model in RAM
                k = int(pet.get("k") or pl.K_HISTORIAS)
                k = max(3, min(60, k))
                # mas historias pedidas -> mas candidatas muestreadas, o no habria de donde sacarlas
                r = limpiar_resultado(P(d[:1200], k_historias=k, n_muestras=max(30, 3 * k)))
                # Las horas van DENTRO del mismo candado y de la misma respuesta: el
                # usuario ya esta esperando, y asi no hay una segunda vuelta.
                if EST is not None:
                    # en /generar solo tiene sentido el previo: el backlog es nuevo y
                    # todavia no hay tareas cerradas ni respuestas que alinear
                    r = EST.estimar(r, **{k: v for k, v in self._anclaje(pet).items()
                                          if k == "previo_equipo"})
            r["segundos_servidor"] = round(time.time() - t, 2)
            self._enviar(200, json.dumps(r, ensure_ascii=False))
        except Exception as e:
            self._enviar(500, json.dumps(dict(error=f"{type(e).__name__}: {e}")))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    # Por omision solo escucha en el propio equipo. Con --lan escucha en todas las
    # interfaces, que es lo que hace falta para entrar desde otra maquina (misma red,
    # o una VPN tipo Radmin/Hamachi). Mientras este asi, la demo es accesible SIN
    # CONTRASENA para cualquiera que alcance esa direccion: se levanta para la ocasion
    # y se apaga al terminar.
    HOST = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
    if HOST != "127.0.0.1":
        import socket
        print(f"  accesible desde fuera en el puerto {PUERTO}:", flush=True)
        try:
            for ip in sorted({i[4][0] for i in socket.getaddrinfo(
                    socket.gethostname(), None, socket.AF_INET)}):
                print(f"    http://{ip}:{PUERTO}", flush=True)
        except Exception:                            # noqa: BLE001
            pass
        print("  (sin contrasena: apagalo cuando termines)", flush=True)
    ThreadingHTTPServer((HOST, PUERTO), Manejador).serve_forever()
