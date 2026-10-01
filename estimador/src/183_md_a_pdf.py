"""Convierte un Markdown con diagramas Mermaid en un PDF, con los diagramas dibujados.

No hace falta instalar nada: usa markdown-it (ya esta en el entorno), la copia local de
mermaid.js y el Chrome que ya hay en la maquina para imprimir.

    python src/183_md_a_pdf.py DISENO_PIPELINE.md
    python src/183_md_a_pdf.py DISENO_PIPELINE.md --salida reports/diseno.pdf

Como funciona:

  1 el Markdown se pasa a HTML, dejando los bloques ```mermaid como <pre class="mermaid">
  2 mermaid.js los dibuja en el navegador, en SVG vectorial
  3 Chrome sin ventana imprime la pagina a PDF

El SVG es vectorial, asi que los diagramas salen nitidos a cualquier zoom en vez de
pixelarse como lo haria un PNG.

Dos cosas que hay que vigilar y por las que el guion no es trivial:

  - mermaid dibuja de forma ASINCRONA. Si Chrome imprime antes de que termine, el PDF
    sale con los diagramas en blanco. Se resuelve con --virtual-time-budget, que hace
    que Chrome adelante su reloj hasta que la pagina se queda quieta.
  - los diagramas altos (el detallado, los C4) se salen de la pagina. Se limitan por
    CSS a la altura util de un A4 y se dejan escalar.
"""
import html as html_mod
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# la copia local de mermaid que se instalo para validar los diagramas
MERMAID = Path(os.environ.get("MERMAID_JS", "")) if os.environ.get("MERMAID_JS") else None
CHROMES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def buscar_mermaid():
    if MERMAID and MERMAID.exists():
        return MERMAID
    p = RAIZ / "node_modules" / "mermaid" / "dist" / "mermaid.min.js"
    if p.exists():
        return p
    # el que se instalo en el directorio de trabajo temporal de la sesion
    base = Path(r"C:\Users\User\AppData\Local\Temp\claude")
    if base.exists():
        for c in base.glob("*/*/scratchpad/node_modules/mermaid/dist/mermaid.min.js"):
            return c
    return None


def buscar_chrome():
    for c in CHROMES:
        if c.exists():
            return c
    return None


def ancla(texto):
    """El mismo identificador que genera GitHub, para que el indice funcione."""
    t = texto.strip().lower()
    t = "".join(c for c in t if c.isalnum() or c in " -_"
                or unicodedata.category(c).startswith("L"))
    return t.replace(" ", "-")


def a_html(md, titulo):
    from markdown_it import MarkdownIt

    mdp = MarkdownIt("commonmark", {"html": True, "linkify": False})
    mdp.enable(["table", "strikethrough"])

    def fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        info = (tok.info or "").strip()
        if info == "mermaid":
            return f'<div class="mermaid-caja"><pre class="mermaid">{html_mod.escape(tok.content)}</pre></div>\n'
        cls = f' class="lang-{info}"' if info else ""
        return f"<pre><code{cls}>{html_mod.escape(tok.content)}</code></pre>\n"

    mdp.add_render_rule("fence", fence)
    cuerpo = mdp.render(md)
    # identificadores en los encabezados, para que los enlaces del indice lleven a algo
    def poner_id(m):
        n, txt = m.group(1), m.group(2)
        plano = re.sub(r"<[^>]+>", "", txt)
        return f'<h{n} id="{ancla(plano)}">{txt}</h{n}>'
    cuerpo = re.sub(r"<h([1-6])>(.*?)</h\1>", poner_id, cuerpo, flags=re.S)
    return cuerpo


CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
* { box-sizing: border-box; }
body { font: 10.5pt/1.5 "Segoe UI", system-ui, sans-serif; color: #1d1c1a;
       background: #fff; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 6pt; line-height: 1.25; page-break-before: always;
     border-bottom: 2px solid #12645d; padding-bottom: 5pt; color: #12645d; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 14pt; margin: 16pt 0 5pt; color: #12645d; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; color: #7d5a17; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
ul, ol { margin: 5pt 0 5pt 0; padding-left: 16pt; }
li { margin: 1.5pt 0; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt;
       background: #f3f1ed; padding: 1px 3px; border-radius: 3px; }
pre { background: #f7f5f2; border: 1px solid #e6e2dc; border-radius: 5px;
      padding: 7pt 9pt; overflow: hidden; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.42;
           white-space: pre-wrap; word-break: break-word; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 9pt;
        page-break-inside: avoid; }
th { background: #12645d; color: #fff; text-align: left; font-weight: 600; }
th, td { border: 1px solid #d9d4cc; padding: 3.5pt 5pt; vertical-align: top; }
tr:nth-child(even) td { background: #faf9f7; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #f7f5f2;
             border-left: 3px solid #7d5a17; page-break-inside: avoid; }
blockquote p { margin: 3pt 0; }
hr { border: 0; border-top: 1px solid #e6e2dc; margin: 12pt 0; }
img { max-width: 100%; height: auto; page-break-inside: avoid; }
a { color: #12645d; text-decoration: none; }
.mermaid-caja { page-break-inside: avoid; text-align: center; margin: 10pt 0; }
.mermaid svg { max-width: 100% !important; width: auto !important;
               max-height: 225mm; height: auto !important; }
/* Un diagrama muy alto encogido para caber en la caja de texto queda diminuto. Los
   que pasan de cierta esbeltez se llevan su propia pagina y usan casi todo el papel:
   el SVG es vectorial, asi que ampliarlo en pantalla no pierde nitidez. */
@page suelta { size: A4; margin: 6mm; }
/* solo salto ANTES: con salto despues tambien, cada diagrama alto dejaba una pagina
   en blanco detras, porque el diagrama ya llena la suya y el texto siguiente empieza
   en la siguiente de todos modos */
.mermaid-caja.alta { page: suelta; page-break-before: always; margin: 0; }
.mermaid-caja.alta svg { max-height: 283mm; }
/* Y al reves: un diagrama muy ancho en pagina vertical solo puede usar 182 mm de
   ancho. En apaisada usa 285, un 57 % mas, que en estos diagramas es la diferencia
   entre leer las etiquetas y no leerlas. */
@page apaisada { size: A4 landscape; margin: 6mm; }
.mermaid-caja.ancha { page: apaisada; page-break-before: always; margin: 0; }
.mermaid-caja.ancha svg { max-height: 196mm; }
"""

PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>__TITULO__</title>
<style>__CSS__</style>
<script src="__MERMAID__"></script>
</head><body>
__CUERPO__
<script>
mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "loose",
                     flowchart: { useMaxWidth: false, htmlLabels: true },
                     themeVariables: { fontFamily: "Segoe UI, system-ui, sans-serif",
                                       fontSize: "14px" } });
(async () => {
  await mermaid.run({ querySelector: ".mermaid" });
  // cada diagrama a la pagina que le conviene por su forma
  const formas = [];
  document.querySelectorAll(".mermaid svg").forEach(svg => {
    const vb = svg.viewBox && svg.viewBox.baseVal;
    if (!vb || !vb.width) return;
    const r = vb.height / vb.width;
    const caja = svg.closest(".mermaid-caja");
    if (r > 1.15) caja.classList.add("alta");        // vertical, hoja propia
    else if (r < 0.62) caja.classList.add("ancha");  // apaisada
    formas.push(r.toFixed(2));
  });
  document.body.setAttribute("data-formas", formas.join(" "));
  document.title = "__TITULO__";
  document.body.setAttribute("data-listo", "1");
})();
</script></body></html>"""


def main():
    libres = [a for a in sys.argv[1:] if not a.startswith("--")]
    origen = Path(libres[0] if libres else "DISENO_PIPELINE.md")
    if not origen.is_absolute():
        origen = RAIZ / origen
    if not origen.exists():
        print(f"no existe {origen}")
        return 1
    if "--salida" in sys.argv:
        destino = Path(sys.argv[sys.argv.index("--salida") + 1])
        if not destino.is_absolute():
            destino = RAIZ / destino
    else:
        destino = RAIZ / "reports" / (origen.stem.lower() + ".pdf")
    destino.parent.mkdir(parents=True, exist_ok=True)

    mer, chrome = buscar_mermaid(), buscar_chrome()
    if mer is None:
        print("falta mermaid.min.js. Instalalo con:  npm install mermaid@11")
        return 1
    if chrome is None:
        print("no se encontro Chrome ni Edge para imprimir")
        return 1
    print(f"mermaid: {mer}")
    print(f"navegador: {chrome.name}")

    t0 = time.time()
    md = origen.read_text(encoding="utf-8")
    n_diag = md.count("```mermaid")
    cuerpo = a_html(md, origen.stem)
    # el HTML se escribe JUNTO al Markdown para que las rutas de las imagenes
    # relativas (reports/figuras_generacion/...) sigan resolviendo
    tmp_html = origen.parent / f".{origen.stem}_pdf.html"
    tmp_mer = origen.parent / ".mermaid.min.js"
    shutil.copyfile(mer, tmp_mer)
    pag = (PAGINA.replace("__CSS__", CSS).replace("__CUERPO__", cuerpo)
           .replace("__MERMAID__", tmp_mer.name)
           .replace("__TITULO__", origen.stem))
    tmp_html.write_text(pag, encoding="utf-8")
    print(f"HTML intermedio: {len(pag) // 1024} KB · {n_diag} diagramas · "
          f"{time.time() - t0:.1f}s", flush=True)

    perfil = RAIZ / ".chrome_pdf_tmp"
    try:
        cmd = [str(chrome), "--headless=new", "--disable-gpu", "--no-sandbox",
               f"--user-data-dir={perfil}",
               "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
               # mermaid dibuja de forma asincrona: sin esto el PDF sale en blanco
               "--virtual-time-budget=60000",
               f"--print-to-pdf={destino}", tmp_html.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if not (destino.exists() and destino.stat().st_size > 20000):
            print("fallo al imprimir:")
            print((r.stderr or "")[-1500:])
            return 1
        print(f"escrito {destino} ({destino.stat().st_size / 1e6:.2f} MB) · "
              f"{time.time() - t0:.0f}s")
        return 0
    finally:
        # pase lo que pase, no dejar basura en el repositorio
        for f in (tmp_html, tmp_mer):
            f.unlink(missing_ok=True)
        shutil.rmtree(perfil, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
