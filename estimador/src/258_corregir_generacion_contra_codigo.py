"""Corrige la Parte A (y la figura B de la Parte 0) contra el codigo del generador.

Al redibujar las figuras contra src/19_pipeline.py, el chat de generacion encontro
tres errores en DISENO_PIPELINE.md (REGENERAR_FIGURAS.md, seccion «Hecho»), los
tres comprobados aqui en el codigo:

  1 los vecinos recuperados NO van al prompt: EJEMPLOS_ARRIBA = EJEMPLOS_ABAJO =
    False (lineas 83-84). Con ellos el sF1 bajaba 0,022 (24_calidad_arriba.py);
    la recuperacion queda para procedencia y orden
  2 el SBERT propio sale de paraphrase-multilingual-mpnet-base-v2
    (06_finetune_sbert.py), no de all-MiniLM-L6-v2
  3 las anclas no salen del plan de clases: salen de las funciones de la
    descripcion (rasgos_con_sujeto) y del plan de ROLES, y se FUSIONAN despues de
    la seleccion (Pipeline._anclas; entran primero, hasta k/2, sin duplicarlas)

Y uno propio: en la figura B el nodo de anclas se llamaba AN, el mismo
identificador que el de Analogia del estimador en ese diagrama, y Mermaid los
fundia en uno. Pasa a ANC.

Ademas: fuera la nota de «figura del 28/09» (las figuras ya estan regeneradas) y
la tabla de hiperparametros gana el segundo modelo de abajo, con los datos de los
registros de entrenamiento que cita el chat de generacion.
"""
import io
from pathlib import Path

DOC = Path("DISENO_PIPELINE.md")


def cambia(s, viejo, nuevo):
    assert s.count(viejo) == 1, f"no esta exactamente una vez: {viejo[:80]!r}"
    return s.replace(viejo, nuevo)


def main():
    s = io.open(DOC, encoding="utf-8").read()

    # ---- Parte 0, figura B
    s = cambia(s, '        V1["Recuperar vecinos · k=2"]',
               '        V1["Recuperar vecinos · k=2<br/>procedencia, no van al prompt"]')
    s = cambia(s, '        AN["ANCLAS · una historia por<br/>función descrita, prefijo forzado"]\n'
                  '        S1["SELECCIÓN por cobertura de rasgos<br/>anclas primero + voraz submodular<br/>+ 6 castigos de forma"]',
               '        ANC["ANCLAS · una historia por<br/>función descrita, prefijo forzado"]\n'
               '        S1["SELECCIÓN por cobertura de rasgos<br/>voraz submodular + 6 castigos de forma"]\n'
               '        FU["FUSIÓN · anclas primero, hasta k/2<br/>y la selección sin duplicarlas"]')
    s = cambia(s, '        V2["Recuperar vecinos por historia"]',
               '        V2["Recuperar vecinos por historia<br/>procedencia, no van al prompt"]')
    s = cambia(s, '        P1 --> P2 --> V1 --> M1 --> B1 --> S1 --> H1 --> V2 --> M2 --> DD\n'
                  '        P2 --> AN --> S1',
               '        P1 --> P2 --> V1 --> M1 --> B1 --> S1 --> FU --> H1 --> V2 --> M2 --> DD\n'
               '        P1 --> ANC --> FU')

    # ---- A.3, figura 1
    s = cambia(s, '    D --> E["Recuperar vecinos del corpus<br/>k=2, tabla o2o"]',
               '    D --> E["Recuperar vecinos del corpus<br/>k=2, tabla o2o<br/>procedencia, no van al prompt"]')
    s = cambia(s, '''    D --> AN["ANCLAS: una historia por función descrita<br/>prefijo forzado, el modelo escribe el beneficio"]
    AN --> H
    G --> H["SELECCIÓN por cobertura de rasgos<br/>las anclas entran siempre, hasta k/2"]
    H --> I["Recuperar vecinos por historia<br/>tabla hist"]''',
               '''    C --> AN["ANCLAS: una historia por función descrita<br/>funciones de la descripción + plan de roles<br/>prefijo forzado, el modelo escribe el beneficio"]
    G --> H["SELECCIÓN por cobertura de rasgos<br/>seleccion_por_rasgos, voraz"]
    H --> FU["FUSIÓN: las anclas primero, hasta k/2<br/>y la selección sin duplicarlas"]
    AN --> FU
    FU --> I["Recuperar vecinos por historia<br/>tabla hist · procedencia"]''')
    s = cambia(s, '''

> La figura exportada (`fig1_pipeline.png`) es del 28/09 y **no lleva las anclas**; el diagrama
> Mermaid de arriba sí está al día. Se regenera con `src/90_figuras.py` del repositorio del
> generador.
''', "\n")

    # ---- A.2.6
    s = cambia(s, """Las anclas entran **siempre**, hasta k/2, y la selección normal rellena el resto.""",
               """Las anclas salen de las funciones de la descripción y del plan de **roles** (no del de
clases), y se **fusionan después** de la selección: entran **siempre**, primero, hasta k/2, y el
resto de lo seleccionado completa el backlog sin duplicarlas.""")

    # ---- A.4
    s = cambia(s, "| 5 | Vecinos de arriba | 2 ejemplos del corpus real por similitud del SBERT propio, que van en el prompt | `_vecinos` |",
               "| 5 | Vecinos de arriba | 2 ejemplos del corpus real por similitud del SBERT propio. **No van en el prompt** (`EJEMPLOS_ARRIBA = False`: con ellos el sF1 bajaba 0,022); quedan para procedencia y orden | `_vecinos` |")
    s = cambia(s, "| 7b | Anclas | una historia por función enumerada, con prefijo forzado y el beneficio elegido entre 6 | `rasgos_con_sujeto`, `capacidad_de_rasgo` |\n"
                  "| 8 | Selección | las anclas primero (hasta k/2); el resto por cobertura de rasgos + 3 términos de reparto + 6 castigos de forma | `seleccion_por_rasgos` |",
               "| 7b | Anclas | una historia por función enumerada, a partir de las funciones de la descripción y del plan de roles, con prefijo forzado y el beneficio elegido entre 6 | `_anclas`, `rasgos_con_sujeto`, `capacidad_de_rasgo` |\n"
               "| 8 | Selección y fusión | cobertura de rasgos + 3 términos de reparto + 6 castigos de forma; después, las anclas primero (hasta k/2) y lo seleccionado sin duplicarlas | `seleccion_por_rasgos`, `__call__` |")
    s = cambia(s, "| 9 | Vecinos de abajo | ejemplos para cada historia elegida | `_vecinos` |",
               "| 9 | Vecinos de abajo | ejemplos para cada historia elegida; **tampoco van en el prompt** (`EJEMPLOS_ABAJO = False`, empate con y sin ellos) | `_vecinos` |")

    # ---- A.5
    s = cambia(s, '        A1["feature -> user story<br/>| role: X | funcion: Y |<br/>&lt;descripción&gt; | examples: &lt;vecinos&gt;"]',
               '        A1["feature -> user story<br/>| role: X | funcion: Y |<br/>&lt;descripción&gt; | examples:"]')
    s = cambia(s, '        B1["decompose story into tasks<br/>&lt;historia&gt; | examples: &lt;vecinos&gt;"]',
               '        B1["decompose story into tasks<br/>&lt;historia&gt; | examples:"]')
    s = cambia(s, """arriba:  feature -> user story | role: courier | funcion: track | <descripción> | examples: <vecinos>
abajo:   decompose story into tasks: <historia> | examples: <vecinos>
```""", """arriba:  feature -> user story | role: courier | funcion: track | <descripción> | examples:
abajo:   decompose story into tasks: <historia> | examples:
```

El campo `examples:` va **vacío** en las dos mitades. Los modelos se entrenaron con ese formato,
pero medido en `24_calidad_arriba.py` los ejemplos recuperados no ayudan: arriba el sF1 sube 0,022
**sin** ellos y no hay fuga de vocabulario de los vecinos; abajo, empate. La recuperación sigue
corriendo para la procedencia (`vecinos_proyecto`) y para ordenar, no para el prompt.""")

    # ---- A.7
    s = cambia(s, '    S0["all-MiniLM-L6-v2"] --> S1["SBERT propio<br/>src/06_finetune_sbert.py<br/>sbert_agix_v1"]',
               '    S0["paraphrase-multilingual-<br/>mpnet-base-v2"] --> S1["SBERT propio<br/>src/06_finetune_sbert.py<br/>sbert_agix_v1"]')
    s = cambia(s, """Hiperparámetros de las dos etapas finales, que son las que hay que repetir:""",
               """Hiperparámetros de los tres modelos desplegados, que son los que hay que repetir:""")
    s = cambia(s, "| abajo (`v5`) | `_hist_agix_v1` | 16 × 1 | 192 | 5e-4 | 4 | **3,3210** · 3,3393 · 3,3565 |",
               "| abajo (`v5`) | `_hist_agix_v1` | 16 × 1 | 192 | 5e-4 | 4 | **3,3210** · 3,3393 · 3,3565 |\n"
               "| abajo, 2.º del conjunto (`_hist_bs8_lr9.2e-4`) | datos: solo `s2s_hist`, 22.871 ejemplos, sin profesor | 8 × 1 | — | 9,2e-4 | parada en la 5.ª | mejor la 3.ª |")
    s = cambia(s, "Se guarda la **mejor época por pérdida de validación**, no la última.",
               "Se guarda la **mejor época por pérdida de validación**, no la última. Datos y épocas leídos de\n"
               "los registros de entrenamiento (`logs/09_v9bt.log`, `logs/44_train_abajo_v5.log`,\n"
               "`logs/09_train_hist_hp.log`) del repositorio del generador: `v9bt` con 25.938 ejemplos, `v5`\n"
               "con 28.274.")

    io.open(DOC, "w", encoding="utf-8").write(s)
    print(f"{DOC}: {len(s):,} caracteres")


if __name__ == "__main__":
    main()
