"""Builds the rating form for the human study as ONE self-contained HTML file.

No server, no internet: the frozen sample is embedded, answers are kept in the
browser while rating, and a button downloads them as JSON to send back. That
survives an advisor opening it on any machine.

The form never reveals which system produced what: the A/B/C key lives in
results/30_estudio_clave.json and is only used by 37_analisis_humano.py.

out: entrega/formulario_estudio.html
"""
import json

BASE = r"F:\Software Task generation Model"
items = json.load(open(f"{BASE}/results/30_estudio_items.json", encoding="utf-8"))
conjuntos = json.load(open(f"{BASE}/results/30_estudio_conjuntos.json", encoding="utf-8"))

datos = json.dumps({"items": items, "conjuntos": conjuntos}, ensure_ascii=False)

HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Estudio de valoración · generación de backlog</title>
<style>
:root{--fondo:#faf9f7;--papel:#fff;--tinta:#1d1c1a;--suave:#6b6762;--linea:#e6e2dc;
--acento:#1e2761;--menta:#02c39a;--radio:10px}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--fondo:#17181a;--papel:#1f2124;
--tinta:#ecebe8;--suave:#a4a09a;--linea:#31343a;--acento:#8fa3e8;--menta:#4fd6b4}}
*{box-sizing:border-box}
body{margin:0;background:var(--fondo);color:var(--tinta);
font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.env{max-width:860px;margin:0 auto;padding:28px 16px 96px}
h1{font-size:1.45rem;margin:0 0 6px}
h2{font-size:1.1rem;margin:32px 0 10px;padding-top:18px;border-top:1px solid var(--linea)}
.sub{color:var(--suave);font-size:.92rem;margin:0 0 18px}
.caja{background:var(--papel);border:1px solid var(--linea);border-radius:var(--radio);
padding:14px 16px;margin:12px 0}
.hist{font-weight:600;margin:0 0 8px;line-height:1.45}
ul{margin:6px 0 10px;padding-left:20px;color:var(--suave);font-size:.9rem}
label{font-size:.88rem;color:var(--suave);margin-right:14px;cursor:pointer}
input[type=radio]{margin-right:4px}
.esc{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0 4px}
.pregunta{font-size:.85rem;color:var(--suave);margin:10px 0 2px}
input[type=text]{width:100%;padding:9px;border:1px solid var(--linea);border-radius:8px;
background:var(--fondo);color:var(--tinta);font:inherit}
button{border:0;border-radius:var(--radio);padding:11px 20px;font:inherit;font-weight:600;
background:var(--acento);color:#fff;cursor:pointer}
.barra{position:fixed;left:0;right:0;bottom:0;background:var(--papel);border-top:1px solid var(--linea);
padding:10px 16px;display:flex;gap:14px;align-items:center;justify-content:center;font-size:.88rem}
.prog{color:var(--suave)}
.bl{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:10px 0}
.bl .caja{margin:0}
.letra{font-weight:700;color:var(--acento);font-size:1.05rem;margin:0 0 6px}
.bl li{font-size:.85rem}
.aviso{background:var(--papel);border:1px solid var(--linea);border-left:3px solid var(--menta);
border-radius:var(--radio);padding:12px 16px;margin:16px 0;font-size:.9rem;color:var(--suave)}
</style></head><body><div class="env">
<h1>Valoración de backlogs generados</h1>
<p class="sub">Tres sistemas generaron historias de usuario y tareas para los mismos proyectos.
Están mezclados y sin identificar: no se sabe cuál es cuál hasta el análisis.
Toma entre 15 y 25 minutos. Las respuestas se guardan en este navegador mientras avanzas.</p>

<div class="caja"><p class="pregunta">Tus iniciales o un apodo (para no repetir evaluadores; no pide datos personales)</p>
<input type="text" id="quien" placeholder="Ej.: MR, asesora, compañero 1"></div>

<div class="aviso">No busques la respuesta &laquo;correcta&raquo;: no existe una única descomposición válida.
Responde como si tuvieras que usar ese backlog para empezar el proyecto.</div>

<h2>Parte 1 · Historias sueltas</h2>
<p class="sub">Para cada historia, con la descripción del proyecto a la vista: ¿es plausible?, ¿la dejarías en el backlog?</p>
<div id="items"></div>

<h2>Parte 2 · Backlogs completos</h2>
<p class="sub">Tres versiones del mismo proyecto, etiquetadas A, B y C en orden aleatorio.</p>
<div id="conjuntos"></div>

<h2>Comentarios</h2>
<div class="caja"><p class="pregunta">Lo que te llamó la atención, bueno o malo (opcional)</p>
<input type="text" id="comentario" placeholder="Opcional"></div>
</div>

<div class="barra">
  <span class="prog" id="prog">0 respuestas</span>
  <button id="descargar">Descargar respuestas</button>
</div>

<script>
const DATOS = __DATOS__;
// El almacenamiento del navegador puede fallar y hay que sobrevivirlo: en ventana
// privada, con los datos de sitio bloqueados o si el fichero se abre como data:
// URL, `localStorage` LANZA en lugar de devolver null. Sin este try el script moria
// en la tercera linea y el formulario salia EN BLANCO, con los titulos puestos y
// ningun item, sin mensaje de error. Las respuestas se entregan con el boton de
// descarga, asi que sin almacenamiento el formulario sigue siendo utilizable: lo
// unico que se pierde es poder cerrarlo y seguir mas tarde.
let HAY_ALMACEN = true;
let R = {};
try { R = JSON.parse(localStorage.getItem("estudio_backlogs") || "{}"); }
catch (e) { HAY_ALMACEN = false; }
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function guardar(){
  if (HAY_ALMACEN) {
    try { localStorage.setItem("estudio_backlogs", JSON.stringify(R)); }
    catch (e) { HAY_ALMACEN = false; aviso(); }
  }
  progreso();
}
function aviso(){
  if ($('aviso_almacen')) return;
  const d = document.createElement('div');
  d.id = 'aviso_almacen';
  d.className = 'caja';
  d.textContent = 'Este navegador no deja guardar el progreso (ventana privada o datos de '
    + 'sitio bloqueados). Puedes responder igual, pero no cierres la pestana hasta '
    + 'descargar las respuestas con el boton del final.';
  document.body.insertBefore(d, document.body.firstChild);
}
function progreso(){
  const n = Object.keys(R).filter(k => k !== "quien" && k !== "comentario").length;
  $('prog').textContent = n + " respuestas de " + (DATOS.items.length*2 + DATOS.conjuntos.length*2);
}

const descripciones = {};
DATOS.conjuntos.forEach(c => descripciones[c.caso] = c.descripcion_es || c.descripcion);

$('items').innerHTML = DATOS.items.map((it, i) => `
  <div class="caja">
    <p class="pregunta">Proyecto: ${esc((descripciones[it.caso]||'').slice(0,150))}</p>
    <p class="hist">${i+1}. ${esc(it.historia_es || it.historia)}</p>
    <ul>${(it.tareas_es||it.tareas||[]).map(t=>`<li>${esc(t)}</li>`).join('')}</ul>
    <p class="pregunta">¿Qué tan plausible es para este proyecto? (1 = sin sentido, 5 = muy plausible)</p>
    <div class="esc">${[1,2,3,4,5].map(v=>`<label><input type="radio" name="p_${it.id}" value="${v}">${v}</label>`).join('')}</div>
    <p class="pregunta">¿La dejarías en el backlog?</p>
    <div class="esc">
      <label><input type="radio" name="k_${it.id}" value="si">Sí</label>
      <label><input type="radio" name="k_${it.id}" value="no">No</label>
      <label><input type="radio" name="k_${it.id}" value="editar">Solo si la edito</label>
    </div>
  </div>`).join('');

$('conjuntos').innerHTML = DATOS.conjuntos.map((c, i) => `
  <div class="caja">
    <p class="pregunta">Proyecto ${i+1}</p>
    <p class="hist">${esc(c.descripcion_es || c.descripcion)}</p>
    <div class="bl">${c.backlogs.map(b=>`
      <div class="caja"><p class="letra">Versión ${b.letra}</p>
      <ul>${(b.historias_es||b.historias).slice(0,8).map(h=>`<li>${esc(h)}</li>`).join('')}</ul></div>`).join('')}
    </div>
    <p class="pregunta">¿Cuál usarías como punto de partida?</p>
    <div class="esc">${c.backlogs.map(b=>`<label><input type="radio" name="mejor_${c.caso}" value="${b.letra}">${b.letra}</label>`).join('')}
      <label><input type="radio" name="mejor_${c.caso}" value="ninguna">Ninguna sirve</label></div>
    <p class="pregunta">¿Cuál se repite más a sí misma?</p>
    <div class="esc">${c.backlogs.map(b=>`<label><input type="radio" name="repite_${c.caso}" value="${b.letra}">${b.letra}</label>`).join('')}
      <label><input type="radio" name="repite_${c.caso}" value="igual">Se repiten igual</label></div>
  </div>`).join('');

document.querySelectorAll('input[type=radio]').forEach(el=>{
  if(R[el.name] === el.value) el.checked = true;
  el.addEventListener('change', ()=>{ R[el.name] = el.value; guardar(); });
});
['quien','comentario'].forEach(id=>{
  if(R[id]) $(id).value = R[id];
  $(id).addEventListener('input', ()=>{ R[id] = $(id).value; guardar(); });
});
progreso();
// si el almacenamiento ya fallo al leer, avisar de entrada: quien evalua tiene que
// saber que no puede cerrar la pestana antes de descargar
if (!HAY_ALMACEN) aviso();

$('descargar').onclick = () => {
  const blob = new Blob([JSON.stringify(R, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'respuestas_' + ((R.quien||'anonimo').replace(/\\W+/g,'_')) + '.json';
  a.click();
};
</script></body></html>"""

salida = f"{BASE}/entrega/formulario_estudio.html"
open(salida, "w", encoding="utf-8").write(HTML.replace("__DATOS__", datos))
print(f"escrito {salida}")
print(f"  {len(items)} historias sueltas y {len(conjuntos)} backlogs completos "
      f"({len(conjuntos) * 3} versiones a ciegas)")
