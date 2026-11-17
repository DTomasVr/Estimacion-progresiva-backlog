"""Extrae un documento autonomo para el chat del GENERADOR de tareas.

El diseno vive unido en DISENO_PIPELINE.md porque son dos tuberias encadenadas,
pero el chat que trabaja la generacion no necesita las tripas del estimador:
necesita su Parte A intacta, mas lo que le toca a EL del enganche, porque
src/34_estimacion.py vive en SU repositorio y ha cambiado.

Se arma: Parte 0 (las dos vistas de la tuberia completa) + Parte A entera +
el enganche + el contrato, con una cabecera que explica el recorte.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, "src")
ORI = Path("DISENO_PIPELINE.md")
SAL = Path("entrega/DISENO_GENERACION.md")

CABECERA = """# Diseño · Tubería de generación de historias y tareas

> **Qué es este documento.** Un recorte de `DISENO_PIPELINE.md`, que une el diseño de las dos
> tuberías encadenadas de PBE. Aquí está todo lo que toca a la **generación**: la vista de
> conjunto, la Parte A íntegra, y las dos secciones del estimador que el generador necesita
> porque el adaptador vive en **este** repositorio.
>
> Lo que **no** está: las tripas del estimador de esfuerzo (arquitectura interna, anclaje,
> bandas, calibración). Eso está en el documento completo, Parte B.

---

"""

NOTA_ENGANCHE = """---

# Lo que el generador necesita del estimador

Estas dos secciones salen de la Parte B del documento completo. Están aquí porque
`src/34_estimacion.py` —el adaptador que une las dos tuberías— vive en el repositorio del
generador, y **ha cambiado**.

> **Cambios recientes que afectan a este repositorio:**
>
> 1. **El factor de estilo es condicional.** `estimar_tareas` ya no lo aplica por omisión; el
>    adaptador tiene que pedirlo con `estilo="generador"`. Aplicarlo a texto escrito por una
>    persona dejaba el sesgo en ×0,82 en vez de 1,0.
> 2. **La banda del total va solo sobre lo que queda abierto**, y se calibra al número de tareas
>    cerradas y al tramo de tamaño, con el cuantil del tramo contraído hacia el global. Antes era
>    ×10,61 fija y no se estrechaba nunca. El adaptador duplicaba la regla antigua y queda
>    alineado.
> 3. El cuantil es el **conforme** (estadístico de orden ⌈(m+1)(1−α)⌉), no el muestral.
> 4. **Estimador v7 (21/10/2026): un solo codificador.** Los dos codificadores ajustados del
>    v6 no aportaban fuera de pliegue y se quitaron; el que queda va en ONNX int8 con el
>    vocabulario podado (397 MB, dentro del estimador, sin caché de HF ni red). El adaptador
>    carga `estimador_pbe_v7.joblib` y cae al v6 si no está. El total del proyecto sale
>    idéntico; las horas **por tarea** sobre texto del generador, ×0,81 las del v6 (abierto:
>    falta medir con tareas del generador con horas reales). Detalle en
>    `ESTIMADOR_V7_INTEGRADO.md` y en la Parte B del documento completo.

---

"""


def main():
    s = io.open(ORI, encoding="utf-8").read()
    # Parte 0 + Parte A: desde el inicio hasta el comienzo de la Parte B
    ini_b = s.index("## B.1 · Qué hace, y en una tabla")
    # el titulo de la Parte B suele ir justo antes; se busca hacia atras
    corte = s.rfind("\n# ", 0, ini_b)
    cuerpo = s[:corte if corte > 0 else ini_b]
    # quitar el titulo original del documento unido, que ya no aplica
    if cuerpo.lstrip().startswith("# "):
        cuerpo = cuerpo[cuerpo.index("\n", cuerpo.index("# ")) + 1:]
    # las dos secciones del enganche
    i1 = s.index("### B.6.6 · Enganche con el generador de backlog")
    f1 = s.index("## B.7 · Contrato de la interfaz")
    i2 = f1
    f2 = s.index("## B.8 · Artefactos y rendimiento medido")
    enganche = s[i1:f1].replace("### B.6.6 ·", "## ·", 1)
    contrato = s[i2:f2].replace("## B.7 · Contrato de la interfaz",
                                "## · Contrato del estimador", 1)
    doc = CABECERA + cuerpo.strip() + "\n\n" + NOTA_ENGANCHE + enganche + contrato
    SAL.parent.mkdir(parents=True, exist_ok=True)
    io.open(SAL, "w", encoding="utf-8").write(doc)
    import re
    print(f"escrito {SAL} · {len(doc):,} caracteres · "
          f"{len(re.findall(r'^## ', doc, re.M))} secciones · "
          f"{doc.count('```mermaid')} diagramas")


if __name__ == "__main__":
    main()
