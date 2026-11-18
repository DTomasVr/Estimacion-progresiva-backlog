"""Pone la Parte A de DISENO_PIPELINE.md al dia con los cambios del generador del 21/10.

El chat de generacion corrigio los cuatro fallos de la prueba de extremo a extremo
(REVISAR_PRUEBA_EXTREMO_A_EXTREMO.md) en src/19_pipeline.py de su repositorio. Las
fuentes de este texto son dos, y solo dos:

  el codigo        F:/Software Task generation Model/src/19_pipeline.py (umbrales,
                   nombres de funcion, comentarios con la medicion de cada cambio)
  su informe       F:/Software Task generation Model/RESPUESTA_REVISION_EXTREMO_A_EXTREMO.md

Las cifras de sF1 de la revision no estan en ningun CSV de results/ de ese
repositorio, solo en el informe; se citan como tales.

Cada reemplazo comprueba que el texto de origen existe exactamente una vez.
"""
import io
from pathlib import Path

DOC = Path("DISENO_PIPELINE.md")
INF = "`RESPUESTA_REVISION_EXTREMO_A_EXTREMO.md`"


def cambia(s, viejo, nuevo):
    assert s.count(viejo) == 1, f"no esta exactamente una vez: {viejo[:80]!r}"
    return s.replace(viejo, nuevo)


def main():
    s = io.open(DOC, encoding="utf-8").read()

    # ---- cabecera e indice
    s = cambia(s, "| 2 × mT5-small podado (121 M) + SBERT propio + Marian | desplegado 28/09/2026 |",
               "| 3 × mT5-small podado (121 M) + SBERT propio + Marian | desplegado 28/09/2026, revisado 21/10/2026 |")
    s = cambia(s, "[A.2 las cinco decisiones](#a2-las-cinco-decisiones-de-diseño)",
               "[A.2 las siete decisiones](#a2-las-siete-decisiones-de-diseño)")
    s = cambia(s, """> La primera tubería: de la descripción del producto al backlog. Estado desplegado a
> 28/09/2026.""", """> La primera tubería: de la descripción del producto al backlog. Estado desplegado a
> 21/10/2026: el diseño del 28/09 más los arreglos de la revisión de extremo a extremo
> (A.2.6, A.2.7 y A.12).""")

    # ---- A.1
    s = cambia(s, """vez. Dos mT5-small podados de 121 M parámetros cada uno; entran de sobra en 8 GB de VRAM.""",
               """vez. Tres mT5-small podados de 121 M parámetros cada uno —uno arriba y dos en conjunto abajo—;
entran de sobra en 8 GB de VRAM. Las anclas de A.2.6 añaden ~0,4 s en GPU y 1,3–2,3 s en CPU.""")
    s = cambia(s, """3. que las tareas pertenezcan a **su** historia y no se repitan.

Las tres se resuelven""", """3. que las tareas pertenezcan a **su** historia y no se repitan;
4. que **cada función que la descripción nombra** esté en el backlog, con su actor.

Las cuatro se resuelven""")

    # ---- A.2
    s = cambia(s, "## A.2 Las cinco decisiones de diseño", "## A.2 Las siete decisiones de diseño")
    s = cambia(s, """un rasgo sin cubrir es trabajo que no existe en el backlog.
""", """un rasgo sin cubrir es trabajo que no existe en el backlog.

**Los rasgos se leen de la descripción entera** (`rasgos_con_sujeto`). El extractor anterior
cortaba por el primer «with» y tiraba la cabeza de la frase, que es justo donde va la función
central: «A web application to book home cleaning services, with…» daba solo lo de detrás del
«with», y «graduates apply with their CV» daba «their CV». La selección llevaba tiempo cubriendo
los rasgos equivocados. Ahora se parte por comas, punto y coma y dos puntos, y cada rasgo guarda
su **sujeto** («companies» en «companies publish offers»), que A.2.6 usa para elegir el actor.
""")
    s = cambia(s, """las tres a favor, con desviación 0,0017.

---

## A.3 Figura 1""", f"""las tres a favor, con desviación 0,0017.

### A.2.6 Una historia ancla por función descrita

En la revisión de extremo a extremo, dos de seis proyectos salían **sin su función central**:
reservar una limpieza, reservar una sala de estudio. No era la selección: en salas el banco traía
**1 candidata de reservar sala entre 28**, y en la ferretería **0 de transferencias entre tiendas**.
Lo que no se genera no se puede elegir. Y pedirlo por la entrada no funciona: con «A mobile app:
find and book study rooms» el modelo sigue escribiendo lecciones, porque la ficha de función son
seis verbos sin objeto y el decodificador copia poco objeto de la entrada.

Lo que el modelo **sí obedece** es un **prefijo forzado** en el decodificador. Por cada función que
la descripción enumera (A.2.4) se construye «As a student, I want to find and book study rooms on
campus in real time, so that» y el modelo escribe solo el beneficio (6 muestras, elige el
reordenador de beneficios). La capacidad sale de `capacidad_de_rasgo`: el rasgo tal cual si empieza
por verbo, o con un verbo ligero si es un sustantivo («low stock alerts» → «receive low stock
alerts»). El actor sale del sujeto del rasgo, con una tabla para los colectivos que el SBERT no
casa con su rol («companies» quedaba más cerca de «graduate» que de «recruiter»).

Las anclas entran **siempre**, hasta k/2, y la selección normal rellena el resto. Es un chequeo de
cobertura, pero en el lado de la **generación**, que es donde estaba el hueco. `PBE_ANCLAR=0` lo
apaga para comparar.

Según el informe del chat de generación ({INF}), de la mejora total de sF1 de la revisión
(+0,0317), **+0,0261 son de las anclas** (ablación, IC95 [+0,018, +0,034]). **Al citarlo:** la
capacidad sale literal de la descripción, que es la entrada y no la referencia, así que no hay
fuga; pero el sF1 premia reutilizar la descripción, y parte de la ganancia es eso.

### A.2.7 El plantel del sector solo entra con respaldo

El umbral de sector (0,65) decide si **hay** plantel, pero no si es el **correcto**. Limpieza a
domicilio caía en *atención domiciliaria* con 0,691 contra 0,682 del sector de limpieza —gana por
0,009— y el backlog salía entero de enfermeras. Salas de estudio nombra solo «estudiante», y el
plantel de docencia metía instructor y asistente, y con ellos asistencia y lecciones. Los sectores
acertados están en 0,75 o más (taller 0,755, clínica 0,827, delivery 0,853, banca 0,920); los
equivocados, entre 0,69 y 0,71.

Por eso, **por debajo de 0,75** (`UMBRAL_PLANTEL_LIBRE`) un rol del plantel solo entra si la
descripción lo **respalda**: comparte con ella una palabra de dominio, por raíz de cinco letras
(«cleaner» con «cleaning», «store manager» con «stores»; `rol_respaldado`). Las palabras de rol que
no dicen nada del dominio —manager, agent, member, administrator…— no cuentan como respaldo.
Calibrado sobre 9 puntos, no más.

Además se leen del español «egresado», «recepción» y, solo en un portal de empleo, «empresa». Al
empleador se le pide *recruiter* y no *employer*: en una sonda de 16 muestras «employer» se obedece
11 veces y el resto sale «lawyer»; «recruiter», 16 de 16.

---

## A.3 Figura 1""")

    # ---- A.3 figura 1
    s = cambia(s, """    B2 --> C["Planificar ROLES<br/>PlanificadorRoles"]""",
               """    B2 --> C["Planificar ROLES<br/>PlanificadorRoles<br/>plantel con respaldo si sector &lt; 0,75"]""")
    s = cambia(s, """    F --> G["Banco de candidatas<br/>filtros de forma y de léxico"]
    G --> H["SELECCIÓN por cobertura de rasgos<br/>seleccion_por_rasgos, voraz"]""",
               """    F --> G["Banco de candidatas<br/>filtros de forma y de léxico"]
    D --> AN["ANCLAS: una historia por función descrita<br/>prefijo forzado, el modelo escribe el beneficio"]
    AN --> H
    G --> H["SELECCIÓN por cobertura de rasgos<br/>las anclas entran siempre, hasta k/2"]""")
    s = cambia(s, """    J --> K["Deduplicar tareas<br/>global 0,90 · intrahistoria 0,95<br/>y 0,95 sin el verbo"]""",
               """    J --> K["Filtrar y deduplicar tareas<br/>no-desarrollo · global 0,90<br/>intrahistoria 0,95 y 0,95 sin el verbo"]""")
    s = cambia(s, """    N --> O["Reparaciones en español<br/>glosario · glosario con fuente<br/>unificar roles · verbo en infinitivo<br/>eco por flexión"]""",
               """    N --> O["Reparaciones en español<br/>glosario · glosario con fuente · glosario general<br/>restos en inglés · unificar roles<br/>verbo en infinitivo · eco por flexión"]""")
    s = cambia(s, """`reasignar_tareas`, `traducir`, `total`.
""", """`reasignar_tareas`, `traducir`, `total`.

> La figura exportada (`fig1_pipeline.png`) es del 28/09 y **no lleva las anclas**; el diagrama
> Mermaid de arriba sí está al día. Se regenera con `src/90_figuras.py` del repositorio del
> generador.
""")

    # ---- A.4
    s = cambia(s, """| 7 | Filtros del banco | formato de historia, rol válido (persona, no sustantivo cualquiera), léxico de Jira | `rol_valido`, `degenerada_texto` |
| 8 | Selección | cobertura de rasgos + 3 términos de reparto + 6 castigos de forma | `seleccion_por_rasgos` |""",
               """| 7 | Filtros del banco | formato de historia, rol válido (persona, no sustantivo cualquiera), léxico de Jira | `rol_valido`, `degenerada_texto` |
| 7b | Anclas | una historia por función enumerada, con prefijo forzado y el beneficio elegido entre 6 | `rasgos_con_sujeto`, `capacidad_de_rasgo` |
| 8 | Selección | las anclas primero (hasta k/2); el resto por cobertura de rasgos + 3 términos de reparto + 6 castigos de forma | `seleccion_por_rasgos` |""")
    s = cambia(s, """| 11 | Deduplicar tareas | 0,90 entre historias · 0,95 dentro · 0,95 **sin el verbo inicial** | `__call__` |""",
               """| 11 | Filtrar y deduplicar tareas | fuera lo que no es desarrollo (verbo como *attend*, *proceed*, *discuss*, *please*; «prepare for»; frases de historia como «I would like») · 0,90 entre historias · 0,95 dentro · 0,95 **sin el verbo inicial** | `tarea_degenerada`, `__call__` |""")
    s = cambia(s, """| 13 | Reasignar | cada tarea va a la historia a la que de verdad pertenece; piso 0,30 | `__call__` |""",
               """| 13 | Reasignar | cada tarea va a la historia a la que de verdad pertenece; piso 0,30. Ninguna historia se queda vacía, y el rescate prefiere una tarea que no esté ya en otra historia | `__call__` |""")
    s = cambia(s, """| 15 | Reparaciones | glosario, glosario con fuente inglesa, unificar nombres de rol, verbo en infinitivo, eco por flexión | `glosario`, `glosario_fuente`, `unificar_roles_es`, `normalizar_verbo_es`, `eco_por_raiz` |""",
               """| 15 | Reparaciones | glosario, glosario con fuente inglesa, **glosario general** disparado por el inglés de origen (stock → existencias, job listing → oferta de empleo, Implement → Implementar, apply → postular en empleo, grade → calificación, appointment → cita, branches → sucursales fuera de un repositorio, test case → caso de prueba), **restos en inglés** (real-time, follow-up, Scheduler, HR…), unificar nombres de rol, verbo en infinitivo, eco por flexión | `glosario`, `glosario_fuente`, `glosario_general`, `unificar_roles_es`, `normalizar_verbo_es`, `eco_por_raiz` |""")

    # ---- A.6 figura 3
    s = cambia(s, """    G -->|sí| H["Plan = actores leídos + plantel del sector<br/>hasta k = 4"]""",
               """    G -->|sí| H["Plan = actores leídos + plantel del sector<br/>hasta k = 4<br/>bajo 0,75: solo roles respaldados"]""")
    s = cambia(s, """- **Si el español nombra 2 actores o más, no se consulta la traducción**, porque su entrada es una
  versión con pérdidas de la misma frase.""", """- **Si el español nombra 2 actores o más, no se consulta la traducción**, porque su entrada es una
  versión con pérdidas de la misma frase.
- **Con similitud de sector por debajo de 0,75, el plantel solo aporta roles respaldados** por la
  descripción (A.2.7). Es la regla que quitó las enfermeras de la limpieza a domicilio.""")

    # ---- A.8
    s = cambia(s, """| `models/..._hist_pbe_v1_v5_pbe_v1` | `09_train_mt5.py` | mitad de abajo |""",
               """| `models/..._hist_pbe_v1_v5_pbe_v1` | `09_train_mt5.py` | mitad de abajo |
| `models/mt5-small-pruned_hist_pbe_v1_bs8_lr9.2e-4` | `09_train_mt5.py` | mitad de abajo, 2.º del conjunto: sola empata, pero se equivoca distinto y la unión sube la cobertura |""")
    s = cambia(s, """| `Helsinki-NLP/opus-mt-{es-en,en-es}` | caché en `models/hf_cache` | traducción de entrada y de salida |""",
               """| `models/marian_es_en`, `models/marian_en_es` | copia local de `Helsinki-NLP/opus-mt-*` | traducción de entrada y de salida. Desde carpeta y no desde el caché de HF: con el caché, estando en línea, la librería volvía a resolver `main` y bajaba otra variante de los pesos (+600 MB, y el paquete dejaba de ser offline sin avisar) |""")

    # ---- A.9
    s = cambia(s, """| `UMBRAL_SECTOR` | 0,65 | a partir de aquí se usa el plantel del sector |""",
               """| `UMBRAL_SECTOR` | 0,65 | a partir de aquí se usa el plantel del sector |
| `UMBRAL_PLANTEL_LIBRE` | 0,75 | por debajo, un rol del plantel solo entra si la descripción lo respalda |
| `ANCLAR` | 1 | una historia ancla por función descrita |
| `N_BEN_ANCLA` | 6 | beneficios muestreados por ancla |
| `CONJUNTO` | `dos_modelos` | la mitad de abajo con los dos modelos; `no` usa uno solo |""")

    # ---- A.12
    s = cambia(s, """| Formato «Como… quiero… para…» | — | **100 %** |
| Atribución tarea→historia | 83 % | **86 %** |
""", f"""| Formato «Como… quiero… para…» | — | **100 %** |
| Atribución tarea→historia | 83 % | **86 %** |

### Revisión de extremo a extremo (21/10/2026)

El chat del estimador corrió el sistema completo sobre seis descripciones y revisó a mano las 199
tareas; los fallos se confirmaron después con **3 semillas** y eran sistemáticos. Cifras del informe
del chat de generación ({INF}), no de un CSV:

| proyecto | antes | después |
|---|---|---|
| Reservas de limpieza | enfermera, cuidador; ninguna reserva | cliente y limpiador; reservar servicios, citas recurrentes, pagos |
| Salas de estudio | instructor, asistente; ninguna sala | «encontrar» y «reservar salas de estudio» siempre; el resto aún tira a lo académico |
| Bolsa de trabajo | faltaban publicar y postular | reclutador «publicar ofertas», egresado «postular con mi CV» |
| Ferretería | sin transferencias; «acciones» | «gestionar transferencias entre tiendas», «existencias» |
| Historias clínicas | bien | igual, y ahora con recepcionista, que la descripción nombra |
| Delivery | bien | publicar menús, pedir, pagar y rutas, cada uno con su actor |

Contra las **328 historias humanas**, pareado en la misma sesión, 3 semillas × 6 proyectos:

| | sF1 |
|---|---|
| antes de la revisión | 0,6354 |
| **después** | **0,6671** |
| diferencia | **+0,0317**, IC95 [+0,025, +0,038] |

Gana en los 6 proyectos y en las 3 semillas; +0,0261 es de las anclas (A.2.6, con su salvedad).
El filtro nuevo de tareas quita 33 de 1.240 (2,7 %) en 36 backlogs, y leídas una a una son todas
ruido. El duplicado literal entre historias es 1 de ~620 tareas (0,2 %).
""")

    # ---- A.13
    s = cambia(s, """| Diversidad de superficie de las descripciones sintéticas |""",
               """| **Fuera de lo que la descripción nombra**, el modelo rellena con lo típico del sector (salas de estudio, fuera de sus dos anclas, sigue tirando a lo académico) | — | abierto. En la interfaz conviene pedir al usuario quién hace qué (`INTEGRACION.md` §4) |
| Beneficios genéricos frecuentes («para poder ofrecer un mejor servicio») | — | abierto |
| Alguna tarea floja bajo las anclas («Crear un proyecto» para «encontrar salas») | — | abierto |
| El **sustantivo** «Procedure for…» pasa el filtro de no-desarrollo, que atrapa el verbo *proceed* («Proceder para solicitudes recurrentes», visto en la prueba del paquete del 21/10) | 1 en 36 tareas de esa prueba | abierto |
| Diversidad de superficie de las descripciones sintéticas |""")

    # ---- Parte 0, figura B: la misma tuberia vista de conjunto
    for viejo, nuevo in [
        ('        P1["Planificar ROLES<br/>4 fuentes, nunca vacío"]',
         '        P1["Planificar ROLES<br/>4 fuentes, nunca vacío<br/>plantel solo con respaldo"]'),
        ('        S1["SELECCIÓN por cobertura de rasgos<br/>voraz submodular + 6 castigos de forma"]',
         '        AN["ANCLAS · una historia por<br/>función descrita, prefijo forzado"]\n'
         '        S1["SELECCIÓN por cobertura de rasgos<br/>anclas primero + voraz submodular<br/>+ 6 castigos de forma"]'),
        ('        M2[["mT5 One2Seq · 121M<br/>haz de 4, sin repetir 4-gramas"]]',
         '        M2[["2 × mT5 One2Seq · 121M en conjunto<br/>haz de 4, sin repetir 4-gramas"]]'),
        ('        DD["Deduplicar · tope de arranques<br/>reasignar cada tarea a su historia"]',
         '        DD["Filtrar no-desarrollo · deduplicar<br/>tope de arranques · reasignar cada tarea"]'),
        ('        P1 --> P2 --> V1 --> M1 --> B1 --> S1 --> H1 --> V2 --> M2 --> DD',
         '        P1 --> P2 --> V1 --> M1 --> B1 --> S1 --> H1 --> V2 --> M2 --> DD\n'
         '        P2 --> AN --> S1'),
        ("selección por cobertura elige 12. La calidad no sale del modelo, sale de la elección.",
         "selección por cobertura elige 12. La calidad no sale del modelo, sale de la elección. Lo único\n"
         "   que se fuerza es que cada función que la descripción nombra tenga su historia (las anclas,\n"
         "   A.2.6), porque lo que el modelo no genera no se puede elegir."),
    ]:
        s = cambia(s, viejo, nuevo)

    io.open(DOC, "w", encoding="utf-8").write(s)
    print(f"{DOC}: {len(s):,} caracteres")


if __name__ == "__main__":
    main()
