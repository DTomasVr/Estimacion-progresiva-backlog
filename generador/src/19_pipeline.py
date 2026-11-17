"""End-to-end Agix generation pipeline, and the thing the bench times.

    project description (es or en)
      -> SBERT encode -> retrieve k neighbour features (upper pool)
      -> mT5 One2One samples N candidate stories -> collapse -> top K
      -> per story: retrieve k neighbour stories-with-tasks (lower pool)
      -> mT5 One2Seq generates the task list (one batch for all stories)
      -> cross-story task de-duplication
      -> provenance: which real projects each suggestion was conditioned on

Every stage is timed separately, because the thesis claim is speed: what costs
what, and what the total is per project.

usage:
  python src/19_pipeline.py "una descripcion de proyecto"      # demo
  python src/19_pipeline.py --demo                             # 3 examples
"""
import collections
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, MT5ForConditionalGeneration

BASE = os.environ.get("AGIX_BASE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# la raiz del proyecto: la carpeta que contiene src/, models/, data/ y results/.
# Se deduce de la ubicacion de este archivo; AGIX_BASE la sobreescribe si hace falta.
SBERT = f"{BASE}/models/sbert_agix_v1"
# destilado 2026-09-25: continuado sobre backlogs de producto escritos por qwen2.5:7b
# (30 sectores). Validado contra 328 historias humanas: 0.5522 -> 0.6504 (results/49_real_30sec.csv)
# Tanda v3/v5 (27/09/2026). Arriba: ficha de rol + planificador, que pone el
# reparto de actores en el rango humano (27,2 % de rol dominante contra 64,2 %
# de la version anterior) y elimina los roles que no son personas. Abajo: datos
# con tope de arranques repetidos, que baja la concentracion de plantilla de
# 65,2 % a 44,3 % en la configuracion desplegada. Comparacion completa en
# entrega/ROLES_Y_COHERENCIA.md; la version anterior en respaldo/.
# Paso 2 (28/09/2026): segunda ficha de control | funcion: Y | al estilo de CTRL,
# con las seis clases agrupadas desde los datos (src/84_clases_funcion.py).
# Obediencia a la ficha 99,0 %. La entropia de clases del backlog pasa de 2,22 a
# 2,39 sobre un maximo de 2,58, con las humanas en 2,45: antes el 38-42 % de las
# historias eran de clase "view" y ahora es el 23 %, contra 22 % humano, y "manage"
# sube del 3 % al 8 % (humanas 6 %). Fidelidad a las 328 historias humanas, tres
# semillas: 0,6351 -> 0,6341, empate dentro del ruido de 0,012.
# El anterior (solo ficha de rol) queda en ..._v3r_agix_v1 y se vuelve a el con
# AGIX_ARRIBA, o se apaga la ficha con AGIX_MODO_FUNCION=0.
# 28/09/2026: v9bt anade al conjunto de v8ff los mismos ejemplos con la descripcion
# retrotraducida EN->ES->EN por los MISMOS Marian que el producto usa en inferencia
# (src/88_retrotraducir_datos.py). Motivo: se entrenaba con ingles escrito a mano y en
# produccion la entrada es ingles de Marian, que cambia ~38 % de las palabras.
# Medido pareado sobre las 328 historias humanas, TRES semillas, las tres a favor:
#   20260926  0,6262 -> 0,6361   (+0,0099)
#   777       0,6217 -> 0,6346   (+0,0129)
#   31337     0,6256 -> 0,6382   (+0,0126)
# media +0,0118 con desviacion 0,0017: el ruido de 0,012 es de los valores absolutos,
# la diferencia pareada es estable. Ademas, sobre los 7 dominios: historias
# casi-duplicadas 10,7 -> 7,1 % (humanas 4,8 %) y atribucion tarea->historia 83 -> 86 %.
# El anterior queda en ..._v8ff_agix_v1 y se vuelve a el con AGIX_ARRIBA.
M_ARRIBA = f"{BASE}/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v9bt_agix_v1"
M_ABAJO = f"{BASE}/models/mt5-small-pruned_hist_agix_v1_v5_agix_v1"
# second trained variant (from the HP search): alone it ties M_ABAJO, but it errs
# differently, so the union of both raises coverage at no precision cost (23_cobertura)
M_ABAJO2 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1_bs8_lr9.2e-4"
# Tercer generador del conjunto: DESACTIVADO. Medido con la seleccion por
# cobertura ya puesta: no aporta NADA de cobertura (87,8 % de rasgos cubiertos y
# 66,7 % de backlogs completos, iguales con y sin el), solo anade 15 tareas por
# backlog sobre rasgos ya cubiertos, y cuesta 6 puntos de anclaje y 4 de
# atribucion. Tareas redundantes inflan el total de horas igual que las que
# faltan lo reducen. Lo que si aportaba era variedad de formulacion (34,1 %
# frente a 42,2 % de concentracion): si algun dia pesa mas eso, se reactiva con
# AGIX_ABAJO3 apuntando a un modelo.
M_ABAJO3 = os.environ.get("AGIX_ABAJO3", "")
CONJUNTO = os.environ.get("AGIX_CONJUNTO", "dos_modelos")   # "no" | "dos_modelos" | "completo"
DISPOSITIVO = os.environ.get("AGIX_DEVICE", "cuda")   # "cpu" to time a GPU-less deployment
# Measured in 24_calidad_arriba.py: the retrieved examples do NOT improve either half
# (upper: +0.022 sF1 on the test epics WITHOUT them and 0 % vocabulary leakage from the
# neighbours; lower: a tie). Retrieval stays for provenance and for ranking, not in the prompt.
EJEMPLOS_ARRIBA = False
EJEMPLOS_ABAJO = False
# Training the upper half on machine-translated Spanish inputs made English worse and
# Spanish no better (26_eval_es.py), so the input is translated at inference instead.
TRADUCIR_ENTRADA = True
_ES = {"de", "la", "que", "el", "en", "los", "las", "para", "con", "una", "por", "del",
       "sus", "sea", "como", "aplicacion", "usuario", "proyecto", "plataforma", "puedan"}
CACHE_EMB = f"{BASE}/data/processed/pipeline_emb.npz"
PAT_HIST = re.compile(r"as\s+(?:an?|the)\s+[^,.]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                      r"(?:want|would\s+like|need|wish|should|must|am\s+able)", re.I)
K_VECINOS = 2
N_MUESTRAS = 30
UMBRAL_DUP = 0.85
K_HISTORIAS = 12
UMBRAL_TAREA_DUP = 0.90         # entre historias distintas
UMBRAL_TAREA_DUP_INTRA = float(os.environ.get("AGIX_DUP_INTRA", 0.95))  # dentro de una historia
# ... y la misma comparacion QUITANDO el verbo inicial. «Crear linea de tiempo» y
# «Anadir linea de tiempo» son la misma tarea: medidos, esos pares dan 0,91 con el
# verbo y 1,00 sin el. Solo se aplica dentro de una historia, donde una casi-copia
# es siempre ruido. Medido sobre la salida desplegada: 7 pares en 256 tareas.
UMBRAL_TAREA_SINV = float(os.environ.get("AGIX_DUP_SINV", 0.95))
BEAMS_ABAJO = int(os.environ.get("AGIX_BEAMS", 4))   # 1 = greedy, the cheap CPU setting
# Sobrescribe por completo la configuracion de generacion de las tareas. Lo usa
# src/64_decodificador.py para barrer decodificadores sin tocar nada mas.
GEN_ABAJO = None
# La mitad de arriba muestrea 30 candidatas con temperatura 1,0, asi que dos
# corridas del MISMO modelo no dan lo mismo. Medido sobre las 328 historias
# humanas: el modelo de produccion puntuo 0,0122 menos al repetir la corrida
# (desvio medio por proyecto 0,0143, maximo 0,0260), que es del tamano de los
# efectos que estamos midiendo. Con AGIX_SEED la medicion es reproducible; sin
# ella el comportamiento es el de antes.
SEMILLA = os.environ.get("AGIX_SEED")


# Machine translation renders product names literally ("helm charts" -> "cartas de
# timon"), so they are masked before translating and restored after: term retention
# went 64 % -> 100 % and fidelity 0.858 -> 0.874 (27_traduccion_salida.py).
TECNICO = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9]*|[a-z]+(?:[A-Z][a-z]+)+|\w+\.\w+|\w+-\w+|\w*\d\w*)\b")
PRODUCTOS = {"helm", "podman", "fabric", "docker", "kafka", "redis", "nginx", "grafana",
             "prometheus", "ansible", "terraform", "jenkins", "kubernetes", "openshift",
             "quarkus", "maven", "gradle", "keycloak", "hadoop", "spark", "kibana",
             "elasticsearch", "istio", "envoy", "postgres", "mongodb"}
_PROD = re.compile(r"\b(" + "|".join(PRODUCTOS) + r")\b", re.I)


def proteger(t):
    fichas = {}

    def rep(m):
        k = f"ZQX{len(fichas)}"
        fichas[k] = m.group(0)
        return k
    return _PROD.sub(rep, TECNICO.sub(rep, t)), fichas


# La traduccion automatica rinde "delivery" como "parto", que en un backlog de
# software nunca es correcto. Se corrige despues de traducir, solo terminos donde
# la acepcion equivocada es imposible en este dominio.
_GLOSARIO = [
    # "delivery" sale como "parto", que en un backlog de software nunca es
    # correcto. OJO: estas cuatro lineas llevaban bytes 0x08 de verdad en lugar
    # de la secuencia \b, asi que el glosario estuvo INERTE desde que se escribio
    # (un 0x08 literal no aparece nunca en el texto traducido). Reparado el
    # 28/09/2026. Es el mismo error que ya pico tres veces en este proyecto: un
    # \\b dentro de una cadena que no es cruda.
    # el articulo cambia de genero: "parto" es masculino y "entrega" femenino
    (r"\bLos partos\b", "Las entregas"), (r"\blos partos\b", "las entregas"), (r"\bEl parto\b", "La entrega"), (r"\bel parto\b", "la entrega"),
    (r"\bUn parto\b", "Una entrega"), (r"\bun parto\b", "una entrega"), (r"\bDel parto\b", "De la entrega"), (r"\bdel parto\b", "de la entrega"),
    (r"\bpartos\b", "entregas"), (r"\bparto\b", "entrega"),
    (r"\bPartos\b", "Entregas"), (r"\bParto\b", "Entrega"),
    # "order" de compra sale como "orden" en la historia y como "pedido" en sus
    # propias tareas, asi que la historia y su trabajo parecen hablar de cosas
    # distintas. Medido sobre las salidas guardadas: 38 "pedido" contra 11
    # "orden", y las once son pedidos de compra ("la disponibilidad de la comida
    # en el orden", "calificar la orden de mi hijo"). El genero cambia, asi que el
    # articulo va dentro del patron, y se excluye el "orden" de secuencia
    # (alfabetico, cronologico, de clasificacion), que si seria legitimo.
    (r"\b[Ll]as \u00f3rdenes\b", "los pedidos"),
    (r"\b[Ll]os \u00f3rdenes\b", "los pedidos"),
    (r"\bLa orden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "El pedido"),
    (r"\bla orden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "el pedido"),
    (r"\bEl orden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "El pedido"),
    (r"\bel orden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "el pedido"),
    (r"\b[Uu]na orden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "un pedido"),
    (r"\b[Dd]e la orden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "del pedido"),
    (r"\b\u00f3rdenes\b", "pedidos"),
    (r"\borden\b(?!\s+(?:alfab|cronol|ascend|descend|de\s+clasificaci))", "pedido"),
    # "recurring bill" -> "ley recurrente". Solo esa locucion: "ley" a secas si
    # puede ser legitima en un backlog de cumplimiento normativo.
    (r"\b[Ll]eyes recurrentes\b", "facturas recurrentes"),
    (r"\b[Ll]ey recurrente\b", "factura recurrente"),
    # "prescription" sale como "prescripcion" (43 veces) y como "receta" (15) para
    # el mismo termino, a veces en la misma historia. "Receta" es lo natural en
    # espanol para una farmacia; "prescripcion" en un backlog de software no es la
    # acepcion juridica, asi que unificar no pierde nada.
    (r"\b[Pp]rescripciones\b", "recetas"),
    (r"\bPrescripci\u00f3n\b", "Receta"),
    (r"\bprescripci\u00f3n\b", "receta"),
    # "dashboard" salia con SIETE traducciones distintas en el mismo backlog.
    # Medido sobre 464 tareas generadas: tablero (34), panel de control (20),
    # tablero de control (6), tablero de herramientas (6), tablero de instrumentos
    # (6), tablero de mando (1) y panel. En ingles solo aparece "dashboard" (54
    # veces) y nunca "board" a secas, asi que unificar no puede pisar un tablero
    # kanban legitimo. El orden importa: primero las locuciones, luego la palabra.
    (r"[Tt]ableros\s+de\s+(?:instrumentos|herramientas|control|mando)", "paneles"),
    (r"[Tt]ablero\s+de\s+(?:instrumentos|herramientas|control|mando)", "panel"),
    (r"[Pp]aneles\s+de\s+control", "paneles"),
    (r"[Pp]anel\s+de\s+control", "panel"),
    (r"\bTableros\b", "Paneles"), (r"\btableros\b", "paneles"),
    (r"\bTablero\b", "Panel"), (r"\btablero\b", "panel"),
]


def glosario(t):
    for pat, rep in _GLOSARIO:
        t = re.sub(pat, rep, t)
    return t


# ----------------------------------- glosario con la FUENTE inglesa como disparador
# El glosario de arriba solo ve el espanol, y hay terminos donde eso no alcanza porque
# la palabra espanola es CORRECTA en otro dominio. Medido sobre las salidas guardadas
# (scratchpad/tecnico_glos.py):
#   «sucursal»  58 apariciones legitimas («gerente de sucursal» = branch manager, en un
#               backlog de banca) y 15 en uno tecnico, donde «versiones de una sucursal»
#               viene de «versions of a branch» y deberia decir «rama».
#   «carrera»   29 vienen de «runs» (ejecuciones de compilacion) y 12 son legitimas,
#               porque salen de «rider» y «organiser» («piloto de carreras»).
# Hacen falta las tres condiciones: que el INGLES traiga el termino disparador, que el
# espanol no este dentro de una locucion de rol, y que la descripcion sea de dominio
# tecnico, porque la sucursal de un banco es una sucursal.
_B = chr(92) + "b"        # ver el aviso de _BB mas abajo: nunca literal
_B_ = _B
_MARCA_TECNICA = re.compile(
    _B + r"(?:continuous integration|build|pipeline|repositor(?:y|ies)|commit|merge|"
    r"deploy|artifact|branch|CI)" + _B, re.I)
# (espanol_malo, ingles_disparador, espanol_bueno, locuciones_que_NO_se_tocan)
_GLOSARIO_FUENTE = [
    (r"sucursales", r"branches", "ramas",
     (r"gerentes? de ", r"directores? de ", r"oficinas? ")),
    (r"sucursal", r"branch", "rama",
     (r"gerente de ", r"director de ", r"oficina ")),
    (r"carreras", r"runs", "ejecuciones",
     (r"piloto de ", r"organizador(?:a)? de ", r"pista de ")),
    (r"carrera", r"run", "ejecución",
     (r"piloto de ", r"organizador(?:a)? de ", r"pista de ")),
    (r"oleoductos|tuberías", r"pipelines", "pipelines", ()),
    (r"oleoducto|tubería", r"pipeline", "pipeline", ()),
]


# Glosario de fuente SIN la puerta tecnica (prueba de extremo a extremo, 21/10/2026).
# Son terminos que el traductor confunde en cualquier dominio, siempre que el ingles de
# origen traiga el disparador: «stock» -> «acciones» en un inventario, «job listing» ->
# «lista de tareas», «Implement» -> «Aplicar». (espanol_malo, disparador_en,
# espanol_bueno, contexto_en_que_lo_anula)
_GLOSARIO_GENERAL = [
    (r"acciones", r"stock", "existencias", r"stock market|shares|stockholder|trading"),
    (r"acción", r"stock", "existencia", r"stock market|shares|stockholder|trading"),
    # el articulo va dentro: «listado» es masculino y «oferta» femenino
    (r"los listados de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "las ofertas de empleo", ""),
    (r"el listado de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "la oferta de empleo", ""),
    (r"un listado de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "una oferta de empleo", ""),
    (r"listados de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "ofertas de empleo", ""),
    (r"listado de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "oferta de empleo", ""),
    (r"listas de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "ofertas de empleo", ""),
    (r"lista de (?:tareas|trabajos?|empleos?)", r"job (?:listing|posting|offer)s?", "oferta de empleo", ""),
    (r"categorías? pendientes?", r"pending grades?", "calificación pendiente", ""),
    (r"categorías", r"grades", "calificaciones", r"categor"),
    (r"categoría", r"grade", "calificación", r"categor"),
    (r"grado", r"grade", "calificación", r"degree|categor"),
    # «branches» fuera de un repositorio son sucursales; dentro, el glosario tecnico
    # de abajo hace lo contrario
    (r"ramas", r"branches", "sucursales", r"git|repositor|commit|merge|code|pull request"),
    (r"rama", r"branch", "sucursal", r"git|repositor|commit|merge|code|pull request"),
    # «appointment» de agenda; el genero cambia, asi que el articulo va dentro
    (r"los nombramientos", r"appointments?", "las citas", r"hire|hiring|position"),
    (r"el nombramiento", r"appointments?", "la cita", r"hire|hiring|position"),
    (r"un nombramiento", r"appointments?", "una cita", r"hire|hiring|position"),
    (r"nombramientos", r"appointments?", "citas", r"hire|hiring|position"),
    (r"nombramiento", r"appointments?", "cita", r"hire|hiring|position"),
    (r"papeles", r"roles", "roles", ""),
    (r"estuches de prueba", r"test cases?", "casos de prueba", ""),
    (r"estuche de prueba", r"test cases?", "caso de prueba", ""),
]
# el verbo de la tarea: «Implement a notification system» -> «Aplicar un sistema ...»
_VERBO_FUENTE = [
    (r"^Aplicar", r"^Implement", "Implementar"),
    (r"^Aplicación de", r"^Implementation of", "Implementación de"),
]
# «apply» en un portal de empleo es postular, no aplicar
_POSTULAR = re.compile("job|offer|cv|resume|vacanc|hiring|recruit", re.I)
# restos que el traductor deja en ingles, con su espanol
_RESTOS_EN = [
    (r"en real-time", "en tiempo real"), (r"real-time", "en tiempo real"),
    (r"follow-ups?", "seguimiento"), (r"[Ss]cheduler", "planificador"),
    (_B_ + "HR" + _B_, "RR. HH."), (r"high-quality", "de alta calidad"),
    (r"check-ins", "registros de entrada"), (r"check-in", "registro de entrada"),
    (r"cold-chain", "de cadena de frío"), (r"well-filled", "bien cubiertas"),
    (r"up-to-(?:fecha|date)", "al día"), (r"high-stock(?:ing)?", "de stock alto"),
    (r"low-stock", "de stock bajo"),
]


def glosario_general(es, en):
    for mal, disp, bien, salvo in _GLOSARIO_GENERAL:
        if not re.search(_B + "(?:" + disp + ")" + _B, en or "", re.I):
            continue
        if salvo and re.search(salvo, en or "", re.I):
            continue

        def _rep(m, bien=bien):
            return bien[0].upper() + bien[1:] if m.group(0)[:1].isupper() else bien

        es = re.sub(_B + "(?:" + mal + ")" + _B, _rep, es, flags=re.I)
    for mal, disp, bien in _VERBO_FUENTE:
        if re.search(disp, (en or "").strip()):
            es = re.sub(mal, bien, es.strip())
    if (re.search(_B + "appl(?:y|ies|ied|ying) (?:to|for|with)" + _B, en or "", re.I)
            and _POSTULAR.search(en or "")):
        es = re.sub(_B + "aplicar a" + _B, "postular a", es)
        es = re.sub(_B + "([Aa])plicar" + _B,
                    lambda m: "Postular" if m.group(1) == "A" else "postular", es)
    for mal, bien in _RESTOS_EN:
        es = re.sub(mal, bien, es)
    return es.replace("en en tiempo real", "en tiempo real")


def glosario_fuente(es, en, descripcion=""):
    """corrige el espanol solo si el ingles de origen trae el termino tecnico Y la
    descripcion es de dominio tecnico. Sin las dos condiciones no se toca nada."""
    if not _MARCA_TECNICA.search(descripcion or ""):
        return es
    for mal, disp, bien, salvo in _GLOSARIO_FUENTE:
        if not re.search(_B + "(?:" + disp + ")" + _B, en or "", re.I):
            continue

        def _rep(m, bien=bien, salvo=salvo):
            antes = es[max(0, m.start() - 24):m.start()].lower()
            if any(re.search(x + "$", antes) for x in salvo):
                return m.group(0)
            return bien[0].upper() + bien[1:] if m.group(0)[:1].isupper() else bien

        es = re.sub(_B + "(?:" + mal + ")" + _B, _rep, es, flags=re.I)
    # «Create a cleaner app» -> «una aplicacion mas limpia»: el limpiador es el ACTOR
    if re.search(_B + "cleaner" + _B, en or "", re.I):
        es = re.sub(r"(aplicación|app|panel|perfil)\s+más\s+limpi[ao]",
                    lambda m: m.group(1) + " del limpiador", es, flags=re.I)
    return es


# Registro de las tareas en espanol. Medido sobre 390 tareas traducidas: el 92 %
# ya empieza en infinitivo ("Almacenar el estado...", "Mostrar el historial...") y
# el 8 % restante sale en imperativo o en pasiva refleja ("Guarde", "Anadase",
# "Agregue", "Preparese"), mas cinco que empiezan por "Por favor". En un mismo
# backlog eso se lee como si lo hubieran escrito dos personas distintas. Son siete
# formas en la muestra, de un punado de verbos; la tabla lleva esas y sus hermanas
# obvias. No es conocimiento del dominio: es conjugacion.
_VERBO_ES = {
    "añádase": "Añadir", "añádanse": "Añadir", "añada": "Añadir", "añadase": "Añadir",
    "añadanse": "Añadir", "añade": "Añadir",
    "agregue": "Agregar", "agréguese": "Agregar", "agreguese": "Agregar",
    "guarde": "Guardar", "guárdese": "Guardar", "guardese": "Guardar",
    "prepárese": "Preparar", "preparese": "Preparar", "prepare": "Preparar",
    "muestre": "Mostrar", "muéstrese": "Mostrar", "muestrese": "Mostrar",
    "cree": "Crear", "créese": "Crear", "creese": "Crear",
    "envíe": "Enviar", "envie": "Enviar", "envíese": "Enviar",
    "permita": "Permitir", "revise": "Revisar", "implemente": "Implementar",
    "actualice": "Actualizar", "notifique": "Notificar", "configure": "Configurar",
    "genere": "Generar", "elimine": "Eliminar", "verifique": "Verificar",
    "registre": "Registrar", "asigne": "Asignar", "valide": "Validar",
    "reduzca": "Reducir", "aplique": "Aplicar", "almacene": "Almacenar",
    "recupere": "Recuperar", "desarrolle": "Desarrollar", "integre": "Integrar",
    "compruebe": "Comprobar", "filtre": "Filtrar", "calcule": "Calcular",
}


# Nominalizaciones. El traductor lee el imperativo ingles como sustantivo y salen
# tareas que no mandan hacer nada: "Preparacion del pedido", "Tiempo de
# visualizacion en el panel". Medido en la revision de cinco dominios: 5 de 184
# tareas (2,7 %). Prefijar "To " al ingles NO sirve, se probo: unas veces arregla y
# otras mete un "Para" delante. Con la tabla deverbal se recupera el infinitivo.
_DEVERBAL = {
    "preparación": "Preparar", "preparacion": "Preparar",
    "visualización": "Mostrar", "visualizacion": "Mostrar",
    "creación": "Crear", "creacion": "Crear",
    "actualización": "Actualizar", "actualizacion": "Actualizar",
    "notificación": "Notificar", "notificacion": "Notificar",
    "validación": "Validar", "validacion": "Validar",
    "configuración": "Configurar", "configuracion": "Configurar",
    "integración": "Integrar", "integracion": "Integrar",
    "implementación": "Implementar", "implementacion": "Implementar",
    "generación": "Generar", "generacion": "Generar",
    "eliminación": "Eliminar", "eliminacion": "Eliminar",
    "recuperación": "Recuperar", "recuperacion": "Recuperar",
    "asignación": "Asignar", "asignacion": "Asignar",
    "revisión": "Revisar", "revision": "Revisar",
    "almacenamiento": "Almacenar", "registro": "Registrar",
    "confirmación": "Confirmar", "confirmacion": "Confirmar",
}
_ART = {"del": "el ", "de la": "la ", "de los": "los ", "de las": "las ", "de": ""}


# Segundo verbo COORDINADO con el infinitivo inicial: «Revisar y vote en la pagina de
# estado». normalizar_verbo_es arregla el primer verbo y el segundo se queda en
# subjuntivo. La regla es deliberadamente estrecha: solo cuando la conjuncion va
# INMEDIATAMENTE despues del infinitivo inicial, porque «Permitir que el usuario
# filtre los pedidos» es subjuntivo correcto y no se debe tocar.
_COORD = {
    "vote": "votar", "revise": "revisar", "valide": "validar", "active": "activar",
    "genere": "generar", "actualice": "actualizar", "elimine": "eliminar",
    "borre": "borrar", "configure": "configurar", "notifique": "notificar",
    "registre": "registrar", "marque": "marcar", "asigne": "asignar",
    "calcule": "calcular", "apruebe": "aprobar", "rechace": "rechazar",
    "filtre": "filtrar", "ordene": "ordenar", "exporte": "exportar",
    "importe": "importar", "publique": "publicar", "archive": "archivar",
    "cancele": "cancelar", "confirme": "confirmar", "descargue": "descargar",
    "guarde": "guardar", "muestre": "mostrar", "cree": "crear", "envie": "enviar",
    "agregue": "agregar", "prepare": "preparar", "permita": "permitir",
}
# La conjuncion puede estar a unas palabras del infinitivo («Crear un informe y envie
# una alerta»), pero entre medias NO puede haber un «que» ni una coma: «Permitir que el
# usuario filtre y ordene los pedidos» lleva dos subjuntivos correctos y no se toca.
_RE_COORD = re.compile(
    r"^([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:ar|er|ir))([^,;]{0,60}?)(\s+(?:y|e|o|u)\s+)"
    r"([a-záéíóúñ]+)\b")
_RE_QUE = re.compile(r"\bque\b|\bpara\b|\bcuando\b|\bsi\b", re.I)


def coordinar_verbo_es(t):
    m = _RE_COORD.match(t or "")
    if not m or _RE_QUE.search(m.group(2)):
        return t
    _sin = str.maketrans("áéíóú", "aeiou")
    inf = _COORD.get(m.group(4).lower().translate(_sin))
    if not inf:
        return t
    return m.group(1) + m.group(2) + m.group(3) + inf + t[m.end():]


def normalizar_verbo_es(t):
    t = re.sub(r"^\s*por\s+favor,?\s*", "", t or "", flags=re.I).strip()
    m = re.match(r"([A-Za-zÁÉÍÓÚÑáéíóúñü]+)(\s|$)", t)
    if not m:
        # la primera palabra va pegada a un signo ("aprobar/rechazar los genericos"):
        # no hay verbo que normalizar, pero la mayuscula inicial si hace falta
        return t[:1].upper() + t[1:] if t else t
    inf = _VERBO_ES.get(m.group(1).lower())
    if inf:
        return coordinar_verbo_es(inf + t[m.end(1):])
    d = re.match(r"([A-Za-z\u00c0-\u017f]+)\s+(de\s+la|de\s+los|de\s+las|del|de)\s+(.+)$", t)
    if d:
        v = _DEVERBAL.get(d.group(1).lower())
        if v:
            return f"{v} {_ART[re.sub(chr(32) + '+', chr(32), d.group(2).lower())]}{d.group(3)}"
    return coordinar_verbo_es(t[:1].upper() + t[1:]) if t else t


_RE_ROL_ES = re.compile(r"^\s*como\s+(?:un[ao]?\s+|el\s+|la\s+|los\s+|las\s+)?"
                        r"([^,]{2,50}?)\s*,", re.I)


def rol_es_de(historia_es):
    m = _RE_ROL_ES.match(historia_es or "")
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None


def unificar_roles_es(historias):
    """Un solo nombre en espanol por actor.

    El traductor alterna: el mismo "support agent" sale como "agente de soporte"
    en una historia y "agente de apoyo" en la siguiente, y el backlog aparenta dos
    actores distintos donde solo hay uno (medido en un backlog de banca:
    5 actores aparentes para 4 reales). La clave del grupo es el rol INGLES, que es
    el que de verdad genero la historia; se queda la variante mas frecuente y las
    demas se reescriben en las historias Y en las tareas.
    """
    grupos = collections.defaultdict(collections.Counter)
    for h in historias:
        en, es = rol_de(h["historia"]), rol_es_de(h.get("historia_es"))
        if en and es:
            grupos[en][es] += 1
    cambios = {}
    for c in grupos.values():
        if len(c) > 1:
            canon = c.most_common(1)[0][0]
            for var in c:
                if var.lower() != canon.lower():
                    cambios[var] = canon
    if not cambios:
        return {}

    def igual_caja(m, canon):
        return canon[0].upper() + canon[1:] if m.group(0)[:1].isupper() else canon

    for var in sorted(cambios, key=len, reverse=True):
        canon = cambios[var]
        pat = re.compile(r"\b" + re.escape(var) + r"\b", re.I)
        for h in historias:
            h["historia_es"] = pat.sub(lambda m: igual_caja(m, canon), h["historia_es"])
            h["tareas_es"] = [pat.sub(lambda m: igual_caja(m, canon), t)
                              for t in h.get("tareas_es", [])]
    return cambios


def restaurar(t, fichas):
    for k, v in fichas.items():
        t = re.sub(rf"\b{k}\b", v, t)
    return t


def es_espanol(t):
    """cheap detector: Spanish function words or accented characters"""
    w = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑ]+", (t or "").lower())
    if not w:
        return False
    acentos = len(re.findall(r"[áéíóúñü¿¡]", t.lower()))
    return (sum(x in _ES for x in w) / len(w) > 0.12) or acentos >= 2


def separar_tareas(texto):
    out, vistos = [], set()
    for t in texto.split(";"):
        t = re.sub(r"\s+", " ", t).strip(" .")
        if len(t) >= 5 and t.lower() not in vistos:
            vistos.add(t.lower())
            out.append(t)
    return out


# ------------------------------------------- bloque Sprint 0: la base tecnica
# Medido el 05/11/2026 sobre 12 backlogs: el generador producia 0 % de tareas
# fundacionales (montar el proyecto, base de datos, acceso, despliegue), contra 7,6 % en
# Jira real y en los datos de entrenamiento. Dos causas: el filtro las tiraba (ver modo
# tecnico arriba), y las 12 historias que se eligen son todas funcionalidades de cara al
# usuario, que la mitad de abajo descompone en su plantilla Almacenar/Mostrar/Notificar.
#
# El bloque va APARTE de las historias de usuario. Tres razones: la interfaz lo puede
# mostrar como Sprint 0; el estimador le pone horas igual; y el sF1 contra las historias
# humanas no se toca, porque las referencias son historias de usuario, no habilitadores.
#
# Las tareas son PLANTILLAS, no salida del modelo. La base tecnica de un proyecto es lo
# mas estandarizado que hay, y ante "configurar el CI/CD" la mitad de abajo responde con
# su plantilla de siempre. Con plantillas el texto es exacto en los dos idiomas y no gasta
# GPU. Lo que el modelo SI decide es poco: que habilitadores tocan, por lo que dice la
# descripcion. Tres van siempre (proyecto, datos, despliegue); el resto se activa por
# indicios en la descripcion inglesa traducida Y en la original en espanol.
SPRINT0 = os.environ.get("AGIX_SPRINT0", "0") == "1"


def _indicio(*palabras):
    return re.compile("|".join(_B + w for w in palabras), re.I)


_HABILITADORES = [
    dict(id="proyecto", siempre=True,
         en="As a developer, I want to set up the project repository and the development "
            "environment, so that the team can start building features on a shared base",
         es="Como desarrollador, quiero configurar el repositorio del proyecto y el entorno "
            "de desarrollo, para que el equipo pueda empezar a construir sobre una base común",
         tareas=[("Create the project repository with its base folder structure",
                  "Crear el repositorio del proyecto con su estructura de carpetas base"),
                 ("Configure the development environment and its dependencies",
                  "Configurar el entorno de desarrollo y sus dependencias"),
                 ("Document how to run the project locally",
                  "Documentar cómo ejecutar el proyecto en local")]),
    dict(id="datos", siempre=True,
         en="As a developer, I want to design the data model and create the database, "
            "so that every feature stores its information consistently",
         es="Como desarrollador, quiero diseñar el modelo de datos y crear la base de datos, "
            "para que cada funcionalidad guarde su información de forma coherente",
         tareas=[("Design the entity-relationship model of the system",
                  "Diseñar el modelo entidad-relación del sistema"),
                 ("Create the database schema and its initial migrations",
                  "Crear el esquema de la base de datos y sus migraciones iniciales"),
                 ("Load sample data for development and testing",
                  "Cargar datos de ejemplo para desarrollo y pruebas")]),
    dict(id="acceso", siempre=False, min_roles=2,
         en_ind=_indicio("account", "login", "log in", "sign up", "register", "user",
                         "role", "access", "permission", "member", "admin", "staff"),
         es_ind=_indicio("cuenta", "usuario", "rol", "acceso", "permiso", "ingres",
                         "registr", "socio", "administrador", "personal"),
         en="As a developer, I want to implement sign-up, login and role-based access, "
            "so that each user only sees what their role allows",
         es="Como desarrollador, quiero implementar el registro, el inicio de sesión y el "
            "acceso por roles, para que cada usuario vea solo lo que su rol permite",
         tareas=[("Implement user sign-up and login",
                  "Implementar el registro y el inicio de sesión de usuarios"),
                 ("Implement password recovery by email",
                  "Implementar la recuperación de contraseña por correo"),
                 ("Implement role-based permissions for each type of user",
                  "Implementar los permisos por rol para cada tipo de usuario")]),
    dict(id="pagos", siempre=False,
         en_ind=_indicio("pay", "payment", "checkout", "card", "subscription", "billing",
                         "invoice", "fee", "deposit", "buy", "purchase"),
         es_ind=_indicio("pag", "cobr", "tarjeta", "suscripcion", "yape",
                         "adelanto", "compra", "tasa"),
         en="As a developer, I want to integrate a payment gateway, so that users can pay "
            "securely inside the app",
         es="Como desarrollador, quiero integrar una pasarela de pagos, para que los usuarios "
            "puedan pagar de forma segura dentro de la aplicación",
         tareas=[("Integrate the payment gateway in test mode",
                  "Integrar la pasarela de pagos en modo de prueba"),
                 ("Handle confirmed and rejected payment responses",
                  "Manejar las respuestas de pago confirmado y rechazado"),
                 ("Record every payment transaction",
                  "Registrar cada transacción de pago")]),
    dict(id="archivos", siempre=False,
         en_ind=_indicio("upload", "photo", "image", "file", "document", "videos", "attach",
                         "cv", "resume", "prescription"),
         es_ind=_indicio("foto", "imagen", "archivo", "document", "videos", "subir", "suben",
                         "adjunt", "cv", "receta"),
         en="As a developer, I want to set up file storage, so that users can upload "
            "documents and images safely",
         es="Como desarrollador, quiero configurar el almacenamiento de archivos, para que "
            "los usuarios puedan subir documentos e imágenes de forma segura",
         tareas=[("Configure the file storage service for uploads",
                  "Configurar el servicio de almacenamiento para los archivos subidos"),
                 ("Validate file type and size on upload",
                  "Validar el tipo y el tamaño de los archivos al subirlos")]),
    dict(id="avisos", siempre=False,
         en_ind=_indicio("notif", "alert", "remind", "sms", "email", "push"),
         es_ind=_indicio("notific", "alerta", "recordatorio", "aviso", "avisa", "correo"),
         en="As a developer, I want to set up the notification service, so that users "
            "receive alerts and reminders on time",
         es="Como desarrollador, quiero configurar el servicio de notificaciones, para que "
            "los usuarios reciban alertas y recordatorios a tiempo",
         tareas=[("Configure the email and push notification service",
                  "Configurar el servicio de notificaciones por correo y push"),
                 ("Create the notification message templates",
                  "Crear las plantillas de los mensajes de notificación")]),
    dict(id="tiempo_real", siempre=False,
         en_ind=_indicio("real time", "real-time", "live tracking", "live location",
                         "live updates", "on the map", "map", "gps", "location", "route"),
         es_ind=_indicio("tiempo real", "en vivo", "seguimiento", "mapa", "ubicacion",
                         "ruta", "recorrido"),
         en="As a developer, I want to set up real-time updates and map support, so that "
            "users see changes and locations without refreshing",
         es="Como desarrollador, quiero configurar las actualizaciones en tiempo real y los "
            "mapas, para que los usuarios vean cambios y ubicaciones sin recargar",
         tareas=[("Set up a real-time channel for live updates",
                  "Configurar un canal en tiempo real para las actualizaciones en vivo"),
                 ("Integrate the maps and geolocation service",
                  "Integrar el servicio de mapas y geolocalización")]),
    dict(id="videollamada", siempre=False,
         en_ind=_indicio("video call", "videocall", "video consultation", "teleconsult",
                         "video meeting"),
         es_ind=_indicio("videoconsulta", "videollamada", "teleconsulta", "video llamada"),
         en="As a developer, I want to integrate a video call service, so that users can "
            "hold remote sessions from the app",
         es="Como desarrollador, quiero integrar un servicio de videollamadas, para que los "
            "usuarios puedan tener sesiones remotas desde la aplicación",
         tareas=[("Integrate the video call service",
                  "Integrar el servicio de videollamadas"),
                 ("Handle camera and microphone permissions",
                  "Gestionar los permisos de cámara y micrófono")]),
    dict(id="reportes", siempre=False,
         # «reports» y no «report»: «players report match results» es un verbo
         en_ind=_indicio("reports", "reporting", "dashboard", "statistic", "analytics",
                         "metric", "kpi", "funnel"),
         es_ind=_indicio("reporte", "informe", "panel", "estadistic", "indicador", "metrica",
                         "embudo"),
         en="As a developer, I want to set up the reporting base, so that reports and "
            "dashboards can query the data they need",
         es="Como desarrollador, quiero preparar la base de reportes, para que los informes "
            "y paneles puedan consultar los datos que necesitan",
         tareas=[("Define the indicators and queries the reports need",
                  "Definir los indicadores y consultas que necesitan los reportes"),
                 ("Create the base reporting module",
                  "Crear el módulo base de reportes")]),
    dict(id="movil", siempre=False,
         en_ind=_indicio("mobile", "android", "ios", "smartphone", "phone"),
         es_ind=_indicio("movil", "celular", "android", "ios", "smartphone"),
         en="As a developer, I want to set up the mobile app project, so that it can be "
            "built and published in the app stores",
         es="Como desarrollador, quiero configurar el proyecto de la aplicación móvil, para "
            "que se pueda compilar y publicar en las tiendas de aplicaciones",
         tareas=[("Create the mobile app project",
                  "Crear el proyecto de la aplicación móvil"),
                 ("Configure the Android and iOS builds",
                  "Configurar las compilaciones para Android e iOS"),
                 ("Prepare the app store listing and publication",
                  "Preparar la ficha y la publicación en las tiendas")]),
    dict(id="despliegue", siempre=True,
         en="As a developer, I want to set up continuous integration and deployment, so "
            "that every change is tested and released safely",
         es="Como desarrollador, quiero configurar la integración y el despliegue continuos, "
            "para que cada cambio se pruebe y se publique de forma segura",
         tareas=[("Configure the CI pipeline to run the tests on every change",
                  "Configurar el pipeline de CI para ejecutar las pruebas en cada cambio"),
                 ("Set up the staging environment",
                  "Configurar el entorno de pruebas previo a producción"),
                 ("Automate the deployment to production",
                  "Automatizar el despliegue a producción")]),
]


# Leidas sobre las 64 descripciones de calibracion: cada una disparaba un habilitador
# que no correspondia. «boletas de pago» son sueldos, no una pasarela; «operadora movil»
# es el negocio, no la plataforma; «publican avisos» son anuncios, no notificaciones;
# «go live» es publicar, no tiempo real.
_FALSOS_INDICIOS = ("boletas de pago", "boleta de pago", "operadora movil", "publican avisos",
                    "avisos clasificados", "go live", "goes live", "are tracked", "is tracked")


def plan_sprint0(descripcion_en, descripcion_original="", roles=()):
    """Los habilitadores que tocan para esta descripcion, en orden de ejecucion: primero
    el proyecto y los datos, despues lo que la descripcion pide, y el despliegue al final.
    Devuelve las historias con la misma forma que las de usuario, mas origen y tipo."""
    en = (descripcion_en or "").lower()
    es = (descripcion_original or "").lower().translate(_ES_TILDE)
    for x in _FALSOS_INDICIOS:
        en, es = en.replace(x, " "), es.replace(x, " ")
    salida = []
    for h in _HABILITADORES:
        if not h["siempre"]:
            por_texto = bool(h["en_ind"].search(en) or h["es_ind"].search(es))
            por_roles = len(roles or ()) >= h.get("min_roles", 99)
            if not (por_texto or por_roles):
                continue
        salida.append(dict(historia=h["en"], historia_es=h["es"],
                           tareas=[a for a, _ in h["tareas"]],
                           tareas_es=[b for _, b in h["tareas"]],
                           origen="habilitador", habilitador=h["id"]))
    # el despliegue al final aunque en el catalogo vaya antes de los opcionales
    salida.sort(key=lambda x: x["habilitador"] == "despliegue")
    return salida


# ------------------------------------------------- validacion del hueco del rol
# La mitad de arriba aprendio la plantilla "As a <rol>, I want ..." pero rellena
# el hueco con cualquier sustantivo de la descripcion, y salen historias como
# "As a service appointment, I want to schedule my first day of repairs".
# El lexico (src/55_lexico_roles.py) sale de las propias fuentes de
# entrenamiento; se acepta el rol si esta listado o si su nucleo es un nucleo de
# persona, asi que "membership manager" pasa y "service appointment" no.
FILTROS = os.environ.get("AGIX_SIN_FILTROS") != "1"   # 1 = apagarlos, para ablacion
_LEX = f"{BASE}/data/processed/roles_lexico.json"
if os.path.exists(_LEX):
    _lex = json.load(open(_LEX, encoding="utf-8"))
    ROLES_OK, CABEZAS_OK = set(_lex["roles"]), set(_lex["cabezas"])
else:                                     # sin lexico el filtro no hace nada
    ROLES_OK, CABEZAS_OK = set(), set()
_RE_ROL = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*[,;]", re.I)


def rol_de(historia):
    m = _RE_ROL.match(historia or "")
    if not m:
        return None
    # el corpus de Jira trae roles pegados en CamelCase ("MembershipManager");
    # se separan antes de normalizar para no rechazar un rol que si es valido
    r = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(1))
    r = re.sub(r"[^a-z\s'-]", " ", r.lower())
    r = re.sub(r"^(a|an|the)\s+", "", re.sub(r"\s+", " ", r).strip())
    return r or None


# ------------------------------------------------- reordenador de beneficios
# El defecto mas grande medido: la clausula "so that" se deduce de la capacidad
# en el 89 % de las historias del profesor y solo en el 34 % de las del modelo.
# No se arregla con mas datos (los datos ya estan bien) sino ELIGIENDO mejor entre
# las 48 candidatas que ya se generan. El juez que sabe distinguirlas es un LLM, y
# en el producto no puede haber ninguno, asi que se destilo en una regresion
# logistica sobre los embeddings del SBERT propio (src/74_reordenador.py):
# AUC 0,728 en descripciones no vistas, y al quedarse con las 12 mejores de 48 la
# coherencia pasa del 53,5 % al 77,3 %.
_RE_TRIPLE = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*,\s*i\s+want\s+(.*?)"
                        r"\s*,\s*so\s+that\s+(.*)$", re.I | re.S)


def partir_historia(h):
    m = _RE_TRIPLE.match(re.sub(r"\s+", " ", h or "").strip())
    return (m.group(1), m.group(2), m.group(3).strip(" .")) if m else None


def cambiar_rol(historia, rol):
    """sustituye el hueco del rol conservando el resto de la historia"""
    m = _RE_ROL.match(historia or "")
    if not m:
        return historia
    art = "an" if rol[:1].lower() in "aeiou" else "a"
    return f"As {art} {rol}," + historia[m.end():]


def rol_valido(historia):
    if not FILTROS or not CABEZAS_OK:
        return True
    r = rol_de(historia)
    if not r:
        return False
    return r in ROLES_OK or r.split()[-1] in CABEZAS_OK


# ------------------------------------------------- tareas degeneradas
# Dos defectos leidos en las salidas: el eco ("Create a store for the store",
# "Add inventory process to the inventory process"), que es el generador
# repitiendose al final de la secuencia, y los restos del corpus de Jira
# ("Upgrade service capacity to 2.0.0-SNAPSHOT", "Add an e2e request").
_PARADA = {"a", "an", "the", "to", "for", "of", "in", "on", "and", "with", "by",
           "from", "at", "as", "their", "its", "my", "all", "new", "each", "per",
           "be", "is", "are", "should", "able", "that", "this", "it"}
_JIRA_PAT = (r"\d+\.\d+(\.\d+)?(-\w+)?|snapshot|\be2e\b|\bosgi\b|\bitest\b"
             r"|\bjavadoc\b|\bmaven\b|\bpom\b|\bjar\b|\bbranch\b|\bchangelog\b"
             r"|\bupgrade\b.*\bversion\b|\bbackport\b")
_JIRA_PAT_NUCLEO = _JIRA_PAT      # solo los patrones escritos a mano; ver modo tecnico
# Los patrones de arriba son doce escritos a mano, y por eso seguian pasando fugas
# como "Create a source distribution for the release" o "Create a config file for
# the orderer" (de donde salia el famoso "ordenador" en espanol). El lexico de
# src/81_lexico_jira_estricto.py son las 167 palabras del lexico de Jira que
# aparecen CERO veces en los 23.247 textos del dominio de producto, asi que vetarlas
# no puede costar una tarea legitima. Medido sobre 302 tareas generadas: toca el
# 4,0 % y al leerlas 11 de 12 son basura ("Reduce the number of tablets in the
# container", "Docs changes", cinco con "orderer").
_LEX_JIRA_EST = []
_p_lex_est = f"{BASE}/data/processed/lexico_jira_estricto.json"
if os.environ.get("AGIX_JIRA_ESTRICTO", "1") == "1" and os.path.exists(_p_lex_est):
    _LEX_JIRA_EST = [w for w in json.load(open(_p_lex_est, encoding="utf-8")) if w.isalpha()]
    _JIRA_PAT += "|" + "|".join(rf"\b{w}\b" for w in _LEX_JIRA_EST)
_JIRA = re.compile(f"({_JIRA_PAT})", re.I)
# Modo TECNICO del filtro (05/11/2026). El lexico estricto son 167 palabras que no
# aparecen nunca en textos de producto de consumo, y por eso vetarlas era seguro en las
# historias de usuario. Pero entre ellas estan justo las del sprint 1: deploy, docker,
# config, install, auth, android, repo, rest, unit, prototype, migrate. Medido: el
# generador producia 0 % de tareas fundacionales contra 7,6 % en Jira, y de 12 tareas
# fundacionales tipicas el filtro tumbaba 6. En el bloque Sprint 0 se usa solo el nucleo
# escrito a mano (versiones, snapshot, maven, backport...), que es ruido en cualquier caso.
_JIRA_NUCLEO = re.compile(f"({_JIRA_PAT_NUCLEO})", re.I)


_FUNCION = {
    # ingles
    "a", "an", "the", "to", "for", "of", "in", "on", "and", "or", "with", "by",
    "from", "at", "as", "that", "this", "it", "is", "are", "be", "can", "so",
    "my", "their", "its", "i", "want", "able", "all", "each", "when", "if",
    # espanol: sin esto el filtro rechaza casi cualquier frase larga, porque
    # "de", "la", "que" o "para" se repiten de forma natural. Era la causa de
    # perder 5 de cada 12 historias traducidas.
    "de", "del", "la", "las", "el", "los", "un", "una", "unos", "unas", "y", "o",
    "que", "para", "por", "con", "en", "se", "su", "sus", "mi", "mis", "al",
    "lo", "como", "quiero", "pueda", "puedan", "poder", "ser", "estar", "sea",
}


_FIJAS = re.compile(r"base de datos|tiempo real|correo electr[oó]nico|"
                    r"punto de venta|palabra clave|database|real time", re.I)
_SUF_RAIZ = ("aciones", "ciones", "iones", "es", "s")


def raiz_es(w):
    """raiz burda, lo justo para que «notificacion» y «notificaciones» coincidan"""
    w = w.lower()
    for suf in _SUF_RAIZ:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)][:7]
    return w[:7]


def eco_por_raiz(t):
    """la misma palabra de contenido dos veces, contando singular y plural como una"""
    pal = re.findall(r"[A-Za-zÁ-Úá-úÑñ][A-Za-zÁ-Úá-úÑñ-]{1,}", _FIJAS.sub(" ", t or ""))
    r = [raiz_es(w) for w in pal if w.lower() not in _FUNCION]
    r = [w for w in r if len(w) >= 5]
    return any(r.count(w) >= 2 for w in set(r))


def degenerada_texto(t):
    """El modo de fallo conocido: una palabra de contenido dando vueltas dentro
    de la misma frase, "Add REST API to the REST API". Las palabras funcionales
    no cuentan: en espanol se repiten por gramatica, no por degeneracion."""
    pal = [w.lower() for w in re.findall(r"[A-Za-zÁ-Úá-úÑñ][A-Za-zÁ-Úá-úÑñ-]{1,}", t or "")]
    pal = [w for w in pal if w not in _FUNCION]
    if any(pal.count(w) >= 3 for w in set(pal)):
        return True
    bi = [tuple(pal[i:i + 2]) for i in range(len(pal) - 1)]
    if any(bi.count(b) >= 2 for b in set(bi)):
        return True
    return ECO_RAIZ and eco_por_raiz(t)


# Relleno de un beneficio: verbos de estado, adverbios y pronombres que pueden formar
# la clausula entera sin decir para que sirve la capacidad.
_RELLENO_BEN = set("""i can be stay keep get make take have has had am is are was
were been being do does did the a an my me our us their its this that these those
to for of in on at by from with without and or but so as it
informed aware updated prepared ready able sure certain safe
quickly rapidly fast soon early properly correctly accurately timely efficiently
effectively easily better best well more less always all any each every when if
them him her they he she we you your""".split())


def beneficio_vacio(ben):
    """el beneficio no aporta ninguna palabra de contenido"""
    return not [w for w in re.findall(r"[a-z]{2,}", (ben or "").lower())
                if w not in _RELLENO_BEN]


def arranque(t):
    """los dos primeros terminos de una tarea, que es la unidad de plantilla"""
    w = re.findall(r"[a-z']+", (t or "").lower())[:2]
    return " ".join(w) if len(w) == 2 else None


# Tope de arranques repetidos por backlog en la SELECCION, no en los datos. Los
# datos de entrenamiento ya estan planos (3,0 % de concentracion en los cinco
# arranques mas frecuentes, contra 2,6 % de las tareas humanas) y el modelo los
# amplifica x14,6 hasta el 43,8 %: es una propiedad del decodificador, no de los
# datos, asi que hay que corregirla en inferencia.
# Baja de 3 a 2 el 28/09/2026, medido en los dos lados. En descripciones de
# producto la concentracion de los cinco arranques mas frecuentes cae del 33,3 % al
# 27,7 % (humanos 2,6 %) sin tocar la atribucion (77,6 % contra 77,8 %). Contra las
# tareas humanas de Jira (300 historias, filtros apagados) cuesta 0,0007 de sF1
# (0,4899 -> 0,4892, nada) y ACERCA el ratio de conteo a las personas: 1,125 ->
# 1,079. El coste es un 13 % menos de tareas, que son justo las terceras
# repeticiones del mismo arranque dentro de un backlog.
ECO_RAIZ = os.environ.get("AGIX_ECO_RAIZ", "1") == "1"
TOPE_ARRANQUE_SALIDA = int(os.environ.get("AGIX_TOPE_SALIDA", 2))
VETAR_ARRANQUES = os.environ.get("AGIX_VETAR", "1") == "1"


# Siglas legitimas de un backlog de producto. Lo que no esta aqui y viene en
# mayusculas de 2 a 5 letras es una clave de proyecto de Jira, que es exactamente
# la forma que tienen ("QE", "AMKT", "NIFI"). Medido sobre 455 tareas generadas:
# solo aparecen dos siglas, "UI" (legitima) y "QE" (la clave), asi que la lista
# blanca no necesita ser larga y el riesgo de tirar una tarea buena es minimo.
SIGLAS_OK = {"UI", "UX", "API", "SMS", "PDF", "CSV", "URL", "ID", "QR", "PIN",
             "OTP", "KYC", "SLA", "FAQ", "GPS", "CRM", "ERP", "POS", "TPV",
             "HTML", "CSS", "JSON", "XML", "HTTP", "HTTPS", "PWA", "IVA", "VAT",
             "RUC", "DNI", "SEO", "CTA", "MFA", "SSO", "TLS", "IBAN", "NFC",
             "RFID", "ETA", "VIP", "ATM", "CVV", "OCR", "ZIP", "GDPR", "PCI",
             "KPI", "B2B", "B2C", "SKU", "EAN", "ISBN", "CEO", "CSS", "SQL"}
# Siglas del trabajo tecnico de base. Fuera del bloque Sprint 0 siguen sin pasar: en una
# historia de usuario de consumo, una sigla desconocida sigue siendo casi siempre una
# clave de proyecto de Jira.
SIGLAS_TECNICAS = {"CI", "CD", "JWT", "REST", "SDK", "DNS", "SSL", "AWS", "GCP", "DB",
                   "ORM", "CDN", "QA", "VPN", "IAM", "RBAC", "SMTP", "APK", "MVP"}


# ------------------------------------------- clases de funcion (paso 2)
# Segunda ficha de control al estilo CTRL: | role: X | funcion: Y |. Las seis clases
# salen de agrupar con el SBERT propio los verbos del medio de 9.077 historias
# (src/84_clases_funcion.py), no de una lista escrita a mano.
#
# Motivo medido: la salida del modelo se concentra en la clase view (42 % de las
# historias, contra 22 % de las humanas y 22 % del profesor) y casi no produce manage
# (2 % contra 6-11 %). La entropia de clases es 2,22 contra 2,45 humana y 2,51 del
# profesor, sobre un maximo de 2,58. O sea que el backlog sale lleno de quiero ver X.
_FUNC = None


def _func():
    global _FUNC
    if _FUNC is None:
        q = f"{BASE}/data/processed/clases_funcion.json"
        _FUNC = (json.load(open(q, encoding="utf-8")) if os.path.exists(q)
                 else {"verbo_clase": {}, "por_familia": {}, "clases": []})
    return _FUNC


def clase_funcion(historia):
    """clase de la funcion que pide la historia, por el primer verbo de su medio"""
    t = partir_historia(historia)
    if not t:
        return None
    for w in re.findall(r"[a-z]+", (t[1] or "").lower()):
        if w not in _FUNCION:
            return _func()["verbo_clase"].get(w)
    return None


def plan_funciones(roles, k, sb=None):
    """reparte clases de funcion entre los actores, para que el banco traiga de
    todas. Se ordena cada clase por P(clase | familia del rol) medido en los dos
    corpus, y se reparte en ronda para que ninguna clase se quede fuera: si solo se
    muestreara la mas probable de cada rol, saldria otra vez todo view."""
    F = _func()
    clases = F.get("clases") or []
    if not roles or not clases:
        return []
    pares = []
    usadas = collections.Counter()
    for r in roles:
        fam = familia_rol_sim(r, sb) if sb is not None else familia_rol(r)
        pes = F.get("por_familia", {}).get(fam or "", {})
        # Solo las clases que ESA familia de rol usa de verdad. Sin este piso, el
        # reparto en ronda pisaba la distribucion medida y le daba funciones de
        # vigilancia a un rol de consumo: salio "Como titular de una cuenta, quiero
        # marcar una alerta de fraude para un usuario", que es trabajo de cumplimiento.
        # P(monitor | consumo) es 0,055, asi que con el piso queda descartada.
        posibles = [c for c in clases if pes.get(c, 0.0) >= PISO_CLASE] or list(clases)
        # entre las posibles, se reparte para que el backlog cubra varias
        orden = sorted(posibles, key=lambda c: (-pes.get(c, 0.0) + 0.25 * usadas[c]))
        for c in orden[:CLASES_POR_ROL]:
            pares.append((r, c))
            usadas[c] += 1
    return pares

# ---------------------------- el rol de consumo no trabaja en la organizacion
# Lo que se quiso hacer y NO funciono: castigar la candidata cuyo VERBO fuera
# improbable para la familia de su rol. La asociacion existe (V de Cramer 0,399 en
# las historias humanas, 0,382 en las del profesor; src/82_rol_funcion.py) pero es
# una TENDENCIA, no una restriccion: marcaba el 15,5 % de las historias y al leerlas
# casi todas eran correctas (un farmaceutico puede buscar recetas, un oficial de
# cumplimiento puede congelar una tarjeta). Medido tambien sobre el beneficio en
# lugar del medio: V = 0,326, igual de flojo.
#
# Lo que SI queda en pie es mucho mas estrecho y esta falsado contra las dos
# poblaciones de referencia: un rol de CONSUMO cuyo beneficio invoca un concepto que
# presupone trabajar en la organizacion. Historias de rol consumidor con eso en el
# beneficio: humanas 1,0 %, profesor 1,1 %, nuestro modelo 8,6 %. Y por palabra:
# workload aparece 12 veces en lo nuestro y CERO en las dos referencias.
#
# inventory queda FUERA de la lista a proposito: el profesor la usa 6 veces con rol
# de comprador y ahi es legitima (un comprador profesional gestiona inventario).
_RE_TRABAJO_AJENO = re.compile(
    r"\b(workload|staffing|headcount|roster|personnel|throughput|"
    r"kpis?|sla|profitability|turnover|my staff|the staff)\b", re.I)
_FAM_NUCLEO = None


def _fam_nucleo():
    global _FAM_NUCLEO
    if _FAM_NUCLEO is None:
        q = f"{BASE}/data/processed/matriz_rol_verbo.json"
        _FAM_NUCLEO = (json.load(open(q, encoding="utf-8")).get("nucleos", {})
                       if os.path.exists(q) else {})
    return _FAM_NUCLEO


_FAM_IDX = None


def _indice_familias(sb):
    """prototipos de cada familia, codificados una vez, para clasificar por vecino"""
    global _FAM_IDX
    if _FAM_IDX is None:
        nuc = _fam_nucleo()
        palabras = sorted(nuc)
        if not palabras:
            _FAM_IDX = ([], [], None)
        else:
            E = sb.encode([f"a {w}" for w in palabras], normalize_embeddings=True,
                          show_progress_bar=False)
            _FAM_IDX = (palabras, [nuc[w] for w in palabras], E)
    return _FAM_IDX


def familia_rol_sim(rol, sb, piso=0.55):
    """Familia del rol por el vecino mas parecido entre los prototipos.

    Hace falta porque la regla de nucleos deja SIN familia al 36 % de los 136 actores
    del plantel (49: account holder, cashier, vet, landlord, underwriter, volunteer...),
    y sin familia ni el castigo del paso 1 ni la ficha de funcion del paso 2 pueden
    actuar. Ampliar la lista a mano seria volver al problema que ya critique dos veces;
    clasificar por similitud con el SBERT propio usa la maquinaria que ya esta cargada y
    cubre el 100 %. Es el esquema de clasificador local por nodo padre de Silla y
    Freitas, con los nucleos conocidos como ejemplos etiquetados.
    """
    directa = familia_rol(rol)
    if directa:
        return directa
    pal, fams, E = _indice_familias(sb)
    if E is None or not rol:
        return None
    import numpy as _np
    q = sb.encode([f"a {rol.strip().lower()}"], normalize_embeddings=True,
                  show_progress_bar=False)[0]
    sims = E @ q
    j = int(_np.argmax(sims))
    return fams[j] if float(sims[j]) >= piso else None


def familia_rol(rol):
    """familia del rol por su nucleo. Si alguna palabra es de una familia de
    TRABAJO manda sobre la de consumo: Staff member tiene por nucleo member, que es
    consumo, y con la regla de la ultima palabra el 3,8 % de las historias humanas
    parecia mal clasificado cuando el error era del clasificador."""
    nuc = _fam_nucleo()
    if not nuc:
        return None
    hallados = [nuc[w] for w in re.findall(r"[a-z]+", (rol or "").lower()) if w in nuc]
    if not hallados:
        return None
    trabajo = [f for f in hallados if f != "consume"]
    return trabajo[0] if trabajo else hallados[-1]

# Prueba de extremo a extremo (21/10/2026): tres formas de «tarea» que no son trabajo
# de desarrollo y el filtro dejaba pasar. Medido sobre 1.240 tareas de 36 backlogs: 33
# caen (2,7 %), y leidas una a una son todas ruido: reuniones de Jira («Please review
# and vote on the delivery date»), verbos que no son de desarrollo («Attend the
# admission of the student») y frases de historia metidas como tarea («I would like
# to be able to choose»). «my»/«me» NO entran: «Add documentation about my records» es
# una tarea legitima de documentacion.
_NO_DESARROLLO = {"attend", "proceed", "be", "participate", "join", "discuss",
                  "consider", "initiate", "enjoy", "wish", "hope", "please"}
_FRASE_HISTORIA = re.compile(
    "(?:" + _B + "i (?:would|'d) like" + _B + "|" + _B + "i want" + _B + "|" + _B
    + "so that i" + _B + "|^as an? " + "|" + _B + "i need to" + _B + ")", re.I)


# La misma idea en forma de SUSTANTIVO, que el filtro del verbo no ve (aviso del chat
# del estimador, 22/10/2026: «Procedure for recurring requests» seguia colandose).
# Se exige la preposicion para no tocar un sustantivo legitimo seguido de su objeto:
# «Meeting room availability» es trabajo real y «Procedure for ...» no lo es nunca.
# Medido sobre 3.061 tareas de las corridas guardadas: caen 21 (0,69 %), las 21 de la
# forma «Procedure for/to», y ninguna de las otras nominalizaciones aparece siquiera.
_NO_DESARROLLO_NOM = re.compile(
    "^(?:procedure|attendance|participation|discussion|consideration|initiation|"
    "enjoyment|proceeding)" + _B + chr(92) + "s+(?:for|to|of|on|in)" + _B, re.I)


def tarea_degenerada(tarea, historia="", tecnica=False):
    if not FILTROS:
        return False
    w0 = (re.findall("[a-z]+", (tarea or "").lower()) or [""])[0]
    if (w0 in _NO_DESARROLLO or _FRASE_HISTORIA.search(tarea or "")
            or _NO_DESARROLLO_NOM.match(tarea or "")
            or re.match("prepare for" + _B, (tarea or "").lower())):
        return True
    p = [w for w in re.findall(r"[a-z]+", (tarea or "").lower()) if w not in _PARADA]
    if len(p) < 2:
        return True
    if (_JIRA_NUCLEO if tecnica else _JIRA).search(tarea):
        return True
    siglas = SIGLAS_OK | SIGLAS_TECNICAS if tecnica else SIGLAS_OK
    if any(s not in siglas for s in re.findall(r"\b[A-Z]{2,5}\b", tarea or "")):
        return True
    # eco: repite el mismo sustantivo en una linea corta
    if len(p) <= 9 and len(p) != len(set(p)):
        return True
    # la tarea es la historia otra vez, sin aportar nada propio
    h = set(re.findall(r"[a-z]+", (historia or "").lower())) - _PARADA
    if h and len(set(p) - h) / len(set(p)) < 0.2:
        return True
    return False


# ------------------------- actores nombrados en la propia descripcion
# Para la poblacion que NO esta en la taxonomia (bodas 0,563 de similitud de
# sector, integracion continua 0,355) el planificador no tiene plantel que
# aplicar, y sin ficha de rol el modelo cae a su habito: 12 roles distintos en 12
# historias, importados del sector entrenado mas parecido (medido en
# entrega/EJEMPLOS_Y_FALLOS.md: la obediencia a la ficha es 100 % con plantel y
# 7 % sin plantel).
#
# Pero la descripcion casi siempre NOMBRA a sus actores: "donde los novios
# contratan proveedores... la lista de invitados". El plantel se saca de ahi con
# el lexico de src/55_lexico_roles.py, que ya se construyo de las tres fuentes de
# entrenamiento (935 roles, 262 nucleos de persona, 114 sustantivos vetados) y no
# a mano.
#
# Forzar el plantel del sector mas parecido cuando NO encaja es peor que no
# forzar: con bodas a umbral 0,55 sale un backlog coherente de restaurante. Por
# eso el plantel ajeno no se usa nunca y el extractor solo actua donde antes se
# devolvia la lista vacia.
_LEXICO = None


def _lexico_roles():
    global _LEXICO
    if _LEXICO is None:
        p = f"{BASE}/data/processed/roles_lexico.json"
        L = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        _LEXICO = (set(L.get("roles", [])), set(L.get("cabezas", [])),
                   set(L.get("no_persona", [])))
    return _LEXICO


def singular(w):
    """plural ingles a singular, lo justo para buscar en el lexico"""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
        return w[:-1]
    return w


# «the cleaning person» daba el rol «person» (calibracion 07/11/2026): son palabras de
# persona que no dicen QUE persona, y como rol no sirven
_SIN_ROL = {"person", "people", "individual", "someone", "somebody", "human", "persona"}


def actores_nombrados(descripcion, k=4):
    """los actores que la descripcion nombra, en orden de aparicion"""
    roles, cabezas, no_per = _lexico_roles()
    if not roles:
        return []
    pal = [singular(w) for w in re.findall(r"[a-z][a-z']*", descripcion.lower())]
    tomado = [False] * len(pal)
    hallado = []                      # (posicion, frase)
    # 1. frases del lexico, de la mas larga a la mas corta, para que "trust and
    #    safety agent" gane a "agent" y consuma sus palabras
    for n in (4, 3, 2):
        for i in range(len(pal) - n + 1):
            if any(tomado[i:i + n]):
                continue
            f = " ".join(pal[i:i + n])
            if f in roles:
                hallado.append((i, f))
                for j in range(i, i + n):
                    tomado[j] = True
    # 2. una sola palabra: del lexico, o cuyo nucleo sea de persona
    for i, w in enumerate(pal):
        if tomado[i] or w in no_per or w in _SIN_ROL:
            continue
        if w in roles or w in cabezas:
            hallado.append((i, w))
            tomado[i] = True
    fuera = []
    for _, f in sorted(hallado):
        if f not in fuera:
            fuera.append(f)
    return fuera[:k]


# ------------------------------------------- paso 4c: los actores, del ESPANOL
# Medido el 28/09/2026. El plan de roles se construia sobre la descripcion ya TRADUCIDA,
# y la traduccion destruye los actores antes de que el planificador los vea:
#   «cuentas para estudiantes, profesores, APODERADOS y administracion»
#       -> «accounts for students, teachers, AGENTS and administration»
#   «avisa al EQUIPO cuando una rama falla»  ->  «alerts the COMPUTER when a branch fails»
# Y es inestable: la misma palabra «apoderado» sale «proxy» en una frase corta y «agent»
# en la descripcion completa, asi que un diccionario sobre el INGLES no puede arreglarlo.
# Se leen los actores del espanol, que es la fuente, y se anteponen al plan.
#
# Sonda de obediencia (scratchpad/sonda.py, 8 muestras por rol, 32 peticiones): el
# generador emite el rol pedido 8/8 en 31 de 32, incluidos «guardian», «vendor» y
# «team member», con y sin los vecinos recuperados en el prompt. O sea que NO habia
# problema de vocabulario: el rol llegaba mal, no se desobedecia. La unica excepcion es
# «tutor», que sale siempre «teacher», y por eso aqui se asigna a «guardian».
_ACTORES_ES = {
    "apoderado": "guardian", "acudiente": "guardian", "tutor": "guardian",
    "equipo": "team member", "personal": "staff member", "plantilla": "staff member",
    "docente": "teacher", "profesor": "teacher", "maestro": "teacher",
    "alumno": "student", "estudiante": "student",
    "director": "principal", "secretaria": "school secretary",
    "mozo": "waiter", "mesero": "waiter", "camarero": "waiter",
    "cocinero": "cook", "chef": "cook", "dueno": "owner", "propietario": "owner",
    "repartidor": "courier", "mensajero": "courier", "conductor": "driver",
    "pasajero": "passenger", "cobrador": "collections agent",
    "encargado": "manager", "responsable": "manager", "supervisor": "supervisor",
    "administrador": "administrator", "operador": "operator",
    "obrero": "worker", "trabajador": "worker", "limpiador": "cleaner",
    "socio": "member", "vecino": "resident", "inquilino": "tenant",
    "arrendatario": "tenant", "huesped": "guest", "invitado": "guest",
    "anfitrion": "host", "novios": "couple", "pareja": "couple",
    "paciente": "patient", "enfermero": "nurse", "medico": "doctor",
    "farmaceutico": "pharmacist", "cajero": "cashier",
    "cliente": "customer", "usuario": "user", "proveedor": "vendor",
    "vendedor": "seller", "comprador": "buyer", "abogado": "lawyer",
    "agricultor": "farmer", "entrenador": "trainer", "estilista": "stylist",
    "moderador": "moderator", "titular": "account holder",
    # prueba de extremo a extremo, 21/10/2026: «egresados» y «recepcion» no se leian
    "egresado": "graduate", "postulante": "candidate", "candidato": "candidate",
    # «empleador» se pide como recruiter: sonda de 16 muestras, «employer» se obedece
    # 11/16 y el resto sale «lawyer»; «recruiter» 16/16. Como con «tutor» arriba.
    "reclutador": "recruiter", "empleador": "recruiter",
    "recepcionista": "receptionist", "recepcion": "receptionist",
    # dominios de la taxonomia v4 (07/11/2026)
    "almacenero": "warehouse clerk", "operario": "machine operator",
    "encuestado": "survey participant", "bodeguero": "shop owner",
    "persona de limpieza": "cleaner", "personal de limpieza": "cleaner",
    "reclutadora": "recruiter", "vendedora": "seller",
}
# «empresa» solo es un actor en un portal de empleo («las empresas publican ofertas»).
# En «una empresa gestiona su flota» es quien encarga el software, no un usuario.
_ES_EMPLEO = ("oferta", "empleo", "postul", "vacante", " cv", "curricul", "egresad")
# «equipo medico» es material, no un equipo de personas; lo mismo «equipo de sonido».
_ES_NO = {"equipo": ("medico", "informatico", "de sonido", "de computo", "de red",
                     "electronico", "de trabajo pesado"),
          # «personal de limpieza» es cleaner, no staff member (va en su propia entrada)
          "personal": ("de limpieza",)}
_ES_TILDE = str.maketrans("áéíóúüñ", "aeiouun")


def actores_es_nombrados(descripcion_es, k=4):
    """actores que la descripcion en espanol nombra, ya traducidos al rol ingles que
    el generador espera. En orden de aparicion, sin repetir."""
    d = " " + descripcion_es.lower().translate(_ES_TILDE) + " "
    hallado = []
    for es, en in _ACTORES_ES.items():
        m = re.search(_BB + es + r"(?:es|s)?" + _BB, d)
        if m is None:
            continue
        cola = d[m.end():m.end() + 24].strip()
        if any(cola.startswith(x) for x in _ES_NO.get(es, ())):
            continue
        if en not in [x for _, x in hallado]:
            hallado.append((m.start(), en))
    if any(x in d for x in _ES_EMPLEO):
        m = re.search(_BB + "empresas?" + _BB, d)
        if m is not None and "recruiter" not in [x for _, x in hallado]:
            hallado.append((m.start(), "recruiter"))
    return [en for _, en in sorted(hallado)][:k]

# ------------------------------------------- paso 4b: el plan nunca queda vacio
# Cuando la descripcion no se parece a ningun sector Y el extractor saca menos de dos
# actores, el codigo dejaba al modelo libre, y libre INVENTA. Medido el 28/09/2026 en
# el dominio de integracion continua (similitud de sector 0,410, un solo actor leido):
# 12 historias con 6 roles fabricados, "hydrator user", "spending collector",
# "volunteer", "hydrator scientist", ninguno del dominio. Nada nombrado es peor que
# cualquier cosa nombrada, asi que el hueco se rellena en tres pasos, del mas fundado
# al mas generico, y solo hasta MIN_ACTORES_SIN_PLANTEL:
#   1. se descartan los candidatos que son MAQUINA y no persona (rol_infraestructura)
#   2. se leen los actores COLECTIVOS que la descripcion nombra ("notifies the team")
#   3. si aun faltan, se completa con roles genericos de cualquier software
# Frontera de palabra construida a mano. Escribir "\b" en una cadena NO cruda
# da el byte 0x08 y el regex queda INERTE sin avisar: ya paso cuatro veces en este
# proyecto (el ultimo, el glosario de traduccion, estuvo muerto desde que se escribio).
_BB = chr(92) + "b"
_COLECTIVOS = {
    "team": "team member", "crew": "crew member", "staff": "staff member",
    "personnel": "staff member", "faculty": "faculty member",
    "committee": "committee member", "board": "board member",
    "administration": "administrator", "management": "manager",
    "leadership": "manager", "operations": "operations manager",
    "kitchen": "kitchen staff",
}
# "school MANAGEMENT system" no nombra a ningun gerente: el colectivo solo cuenta si no
# es el modificador de un sustantivo de producto.
_COL_PRODUCTO = re.compile(
    r"^\s+(?:system|software|platform|tool|app|module|dashboard|suite|service)\b")
_VERBOS_INFRA = (r"run|runs|running|ran|execute|executes|executing|deploy|deploys|"
                 r"deployed|host|hosts|hosted|install|installs|provision|provisions")


def actores_colectivos(descripcion, k=4):
    """actores colectivos que la descripcion nombra, ya en singular de persona"""
    d = descripcion.lower()
    hallado = []
    for w, r in _COLECTIVOS.items():
        m = re.search(_BB + w + _BB, d)
        if m is None or _COL_PRODUCTO.match(d[m.end():]):
            continue
        if r not in [x for _, x in hallado]:
            hallado.append((m.start(), r))
    return [r for _, r in sorted(hallado)][:k]


def rol_infraestructura(descripcion, cand):
    """el candidato es una MAQUINA, no una persona: aparece SOLO como objeto de
    «on/onto» colgado de un verbo de ejecucion u hospedaje ("runs build pipelines on
    agents"). Falsificada sobre las 622 descripciones de la taxonomia: descarta 0 de
    1.507 actores, y 1 de 14 en los siete dominios del producto, que es el que se
    buscaba. Si la palabra aparece tambien en otra posicion es persona y no se toca."""
    d = descripcion.lower()
    n = re.escape(cand)
    pat = re.compile(_BB + "(?:" + _VERBOS_INFRA + ")" + _BB
                     + r"[^.,;]{0,40}?" + _BB + r"on(?:to)?\s+"
                     + r"(?:the|a|an)?\s*" + n + "s?" + _BB)
    if not pat.search(d):
        return False
    pos = [m.start() for m in re.finditer(_BB + n + "s?" + _BB, d)]
    dentro = {o for m in pat.finditer(d) for o in pos if m.start() <= o < m.end()}
    return len(dentro) == len(pos)


ROLES_GENERICOS = [r.strip() for r in os.environ.get(
    "AGIX_ROLES_GENERICOS", "administrator,user,support agent").split(",") if r.strip()]

# ------------------------------------------------- planificador de roles
# Segundo metodo contra el defecto del rol. El primero (rol_valido) DESCARTA las
# candidatas malas; este no deja que ocurran: decide primero QUIEN es el actor y
# se lo pasa al generador como ficha de control
#   feature -> user story | role: shopper | <descripcion> | examples:
# El plantel sale del sector mas parecido de la taxonomia v3 (40 sectores, 136
# actores escritos a mano), y si la descripcion nombra un actor del lexico, ese
# va primero. Asi el rol es valido POR CONSTRUCCION y el backlog tiene varios
# actores sin depender de la suerte del muestreo.
FICHA_ROL = os.environ.get("AGIX_FICHA_ROL") == "1"
# Si la descripcion no se parece a ningun sector de la taxonomia, el plantel
# seria de otro dominio ("As a parking inspector" para un servicio OAuth2), asi
# que mejor no forzar y dejar que trabaje el validador. Medido sobre 8
# descripciones: cuando el sector acierta la similitud es 0,79-0,84 y cuando se
# equivoca 0,34-0,62, asi que 0,70 las separa.
# Medido el 27/09/2026 (scratchpad/banda.py): de 21 descripciones de producto, 16 pasan
# de 0,70 y 3 caen en [0,65 0,70) con el sector CORRECTO en el puesto 1 (limpieza 0,678,
# ticketera 0,695, colegio 0,674). Las 2 que estan de verdad fuera de la taxonomia se
# quedan fuera (0,552 y 0,613), que es lo que se quiere: forzar un plantel ajeno es peor
# que no forzar ninguno (probado con bodas a 0,55: sale un backlog de restaurante).
# Sin plantel la obediencia al rol es 3/42 = 7 %; con plantel, 34/34 = 100 %.
UMBRAL_SECTOR = float(os.environ.get("AGIX_UMBRAL_SECTOR", 0.65))
# Reparar el rol en vez de descartar la candidata: no hace falta reentrenar.
REPARAR_ROL = os.environ.get("AGIX_REPARAR_ROL") == "1"
EXTRAER_ROLES = os.environ.get("AGIX_EXTRAER_ROLES", "1") == "1"
UNIF_ROL_ES = os.environ.get("AGIX_UNIF_ROL_ES", "1") == "1"
REPARAR_ROL_TAREA = os.environ.get("AGIX_REPARAR_ROL_TAREA", "1") == "1"
REPARAR_VERBO_TAREA = os.environ.get("AGIX_REPARAR_VERBO", "1") == "1"
# castigo por la forma de la historia; ver Pipeline._pena_forma
NU_CIRCULAR = float(os.environ.get("AGIX_NU_CIRC", 0.50))
NU_LARGO = float(os.environ.get("AGIX_NU_LARGO", 0.20))
TOPE_PALABRAS_CAP = int(os.environ.get("AGIX_TOPE_PAL_CAP", 18))
NU_AUTO = float(os.environ.get("AGIX_NU_AUTO", 0.40))
NU_ECO = float(os.environ.get("AGIX_NU_ECO", 0.40))
NU_ROLFUN = float(os.environ.get("AGIX_NU_ROLFUN", 0.45))
# Beneficio VACIO: la clausula «so that» no aporta ningun contenido propio, solo
# relleno («so that I can stay informed»). Criterio deliberadamente estrecho: cero
# palabras de contenido, no una. Medido sobre 84 historias: marca 3 (3,6 %) y las
# tres son vacias de verdad. Con el umbral en UNA palabra marcaria 9 mas, y entre
# ellas «so that I can make informed decisions» y «so that I can be aware of any
# issues», que son beneficios legitimos; por eso el corte esta en cero.
# NO cubre el otro defecto de beneficio que se lee (el objeto ajeno al dominio,
# «informed decisions about the spending» en un colegio): ese es la fuga de sector
# del paso 3, cerrada como negativo medido.
NU_VACIO = float(os.environ.get("AGIX_NU_VACIO", 0.45))
# paso 2: ficha de funcion. Se activa sola si el modelo de arriba se entreno con ella
# (los conjuntos v8ff), igual que la ficha de rol se decide por modelo y no por entorno.
MODO_FUNCION = os.environ.get("AGIX_MODO_FUNCION", "auto")
CLASES_POR_ROL = int(os.environ.get("AGIX_CLASES_ROL", 2))
MU_CLASE = float(os.environ.get("AGIX_MU_CLASE", 0.12))
PISO_CLASE = float(os.environ.get("AGIX_PISO_CLASE", 0.10))
# castigo por capacidad repetida por actor; ver seleccion_por_rasgos
# Castigo por capacidad repetida por actor. APAGADO por omision: medido el
# 28/09/2026 y descartado. Funciona (los casi-duplicados pasan de 7,1 % a 0,0 %
# y la atribucion de 86 % a 89 %) pero CUESTA fidelidad a las historias humanas,
# -0,0070 de sF1 con tres semillas, las tres en contra, y 12 de los 18 pares
# proyecto-semilla a la baja. Tres razones para no desplegarlo:
#   1. el efecto se SATURA: 0,15 y 0,25 dan exactamente el mismo sF1, asi que el
#      coste no es del peso sino de quitar la historia;
#   2. 0,0 % de casi-duplicados esta por DEBAJO del 4,8 % humano, o sea que pasa de
#      largo el blanco; y la version con cupo, que apunta a igualar esa tasa, es
#      inoperante porque nuestros duplicados ya llegan como un par por backlog;
#   3. la subida de atribucion es justo el artefacto conocido: la atribucion mejora
#      al empobrecer el backlog, asi que aqui no es prueba de nada.
# El mecanismo queda para experimentar con AGIX_NU_CAP.
NU_CAP = float(os.environ.get("AGIX_NU_CAP", 0.0))
# CUPO: cuantas repeticiones de capacidad por actor se toleran ANTES de castigar.
# Las personas repiten: en los 6 proyectos reales, el 4,8 % de las historias de un
# backlog de 12 tiene un casi-duplicado, o sea 0,6 historias. Minimizar a cero deja
# el backlog MENOS parecido al humano que dejarlo como estaba, asi que el blanco es
# igualar esa tasa, no bajarla del todo.
CUPO_CAP = int(os.environ.get("AGIX_CUPO_CAP", 0))
PISO_CAP = float(os.environ.get("AGIX_PISO_CAP", 0.75))
REASIGNAR_TAREAS = os.environ.get("AGIX_REASIGNAR", "1") == "1"
MARGEN_REASIGNAR = float(os.environ.get("AGIX_MARGEN_REASIG", 0.10))
PISO_TAREA = float(os.environ.get("AGIX_PISO_TAREA", 0.30))
# sector: el plantel de la taxonomia manda y el extractor solo cubre el hueco.
# descripcion: manda lo que la descripcion NOMBRA, y el plantel del sector solo
# rellena hasta k. Medido sobre los 6 proyectos reales contra los actores que
# usaron las personas (results/80_extractor.csv): extractor 0,847 de precision y
# 0,765 de cobertura, plantel del sector 0,250 y 0,178.
# En el producto (49_eval_real, 328 historias humanas): 0,6285 -> 0,6456 con la
# semilla 20260926 y 0,6137 -> 0,6330 con la 777. Por eso "descripcion" es la
# omision desde el 28/09/2026; AGIX_MODO_ROLES=sector vuelve al anterior.
MODO_ROLES = os.environ.get("AGIX_MODO_ROLES", "descripcion")
# paso 4a (28/09/2026): CON plantel basta UN actor nombrado para que entre en el plan.
# La regla anterior pedia dos, y por eso "customers" no llegaba nunca a un backlog de
# banca ni "workers" a uno de limpieza, aunque la descripcion los nombra: el extractor
# saca un solo actor en 3 de 7 dominios y ahi se descartaba entero. El motivo del dos
# era evitar el backlog de un unico actor repetido, y eso solo puede pasar cuando el
# extractor es la UNICA fuente; con plantel los otros k-1 roles lo diluyen. Sin plantel
# siguen haciendo falta dos (MIN_ACTORES_SIN_PLANTEL).
MIN_ACTORES_CON_PLANTEL = int(os.environ.get("AGIX_MIN_ACTORES_PLANTEL", 1))
MIN_ACTORES_SIN_PLANTEL = int(os.environ.get("AGIX_MIN_ACTORES_SOLO", 2))
# Prueba de extremo a extremo (21/10/2026, 6 proyectos x 3 semillas): el umbral decide
# si HAY plantel, pero no si es el CORRECTO. Limpieza a domicilio cae en atencion
# domiciliaria por 0,009 sobre el sector de limpieza y sale entero de enfermeras; salas
# de estudio nombra solo «estudiante» y el plantel de docencia mete instructor y
# asistente, y con ellos asistencia y lecciones. Los sectores acertados estan en 0,75 o
# mas (taller 0,755, clinica 0,827, delivery 0,853, banca 0,920); los equivocados en
# 0,69-0,71. Por debajo de UMBRAL_PLANTEL_LIBRE un rol del plantel solo entra si la
# descripcion lo RESPALDA: comparte con ella una palabra de dominio («cleaner» con
# «cleaning», «store manager» con «stores»). Calibrado sobre 9 puntos, no mas.
UMBRAL_PLANTEL_LIBRE = float(os.environ.get("AGIX_UMBRAL_PLANTEL_LIBRE", 0.75))
# palabras de rol que no dicen nada del dominio: no cuentan como respaldo
_ROL_VACIO = {"manager", "agent", "member", "administrator", "admin", "user", "assistant",
              "coordinator", "owner", "operator", "staff", "officer", "specialist",
              "lead", "head", "representative", "support", "team", "senior", "junior"}


_TRANSACCION = re.compile(_BB + "(?:book|books|booking|order|orders|buy|buys|purchase|"
                          "pay|pays|payment|payments|reserve|reservation)" + _BB)
# «production orders», «purchase orders», «raw material per order» no son una compra de
# un cliente. Calibracion 07/11/2026: en una fabrica de muebles la regla metia «cliente»
# en 4 de 12 historias. Se quitan antes de buscar senales de compra.
_ORDEN_NO_COMPRA = re.compile("(?:production|purchase|work|manufacturing|supplier|service|"
                              "transfer|maintenance|repair) orders?|(?:per|each|in) order",
                              re.I)


def _hay_transaccion(texto):
    return bool(_TRANSACCION.search(_ORDEN_NO_COMPRA.sub(" ", (texto or "").lower())))


_CONSUMO = {"customer", "client", "buyer", "shopper", "guest", "patient", "passenger",
            "student", "member", "tenant", "candidate", "graduate", "user"}


def _raices(texto):
    return {w[:5] for w in re.findall("[a-z]+", texto.lower()) if len(w) >= 4}


def rol_respaldado(descripcion_en, rol):
    """True si alguna palabra de dominio del rol aparece en la descripcion (por raiz
    de cinco letras, para que «cleaner» case con «cleaning»)."""
    d = _raices(descripcion_en)
    return any(w[:5] in d for w in re.findall("[a-z]+", rol.lower())
               if len(w) >= 4 and w not in _ROL_VACIO)


# ------------------------------------------------- seleccion submodular
# Las dos mitades devuelven CONJUNTOS, y hoy los eligen con "colapsar
# casi-duplicadas y quedarse con las k mejores por relevancia". Eso optimiza
# relevancia INDIVIDUAL y castiga la redundancia despues, que es por lo que se
# perdian historias al filtrar.
#
# La formulacion correcta es cobertura de conjunto. Con V el banco de candidatas,
#     F(S) = sum_{v in V} max_{s in S} sim(v, s)
# es la funcion de LOCALIZACION DE INSTALACIONES: monotona y submodular, asi que
# el voraz da (1 - 1/e) del optimo. Tres cosas salen gratis de ahi:
#
#   - la redundancia se penaliza por construccion (ganancias decrecientes), asi
#     que no hace falta deduplicar aparte;
#   - la preferencia por CONSENSO que hace util la decodificacion MBR ya esta
#     dentro: una candidata cercana a muchas otras cubre mas de V, luego se
#     prefiere sola. No hay que elegir entre MBR y cobertura;
#   - el TAMANO del conjunto lo puede decidir el contenido: se corta cuando la
#     ganancia marginal baja de una fraccion de la primera.
#
# Referencia de la formulacion: Lin y Bilmes (2011), seleccion submodular para
# resumen extractivo.
SUBMODULAR = os.environ.get("AGIX_SUBMODULAR") == "1"
# Medido en results/68_cobertura*: sin cuota por rol los backlogs que cubren
# TODOS los rasgos de la descripcion pasan del 53,3 % al 66,7 %. El precio es
# que el rol dominante sube de 26,9 % a 45,8 %, que sigue dentro del rango
# humano (30-48 %). La cuota nos hacia mas diversos que las personas y lo
# pagabamos en cobertura del alcance. AGIX_BANCO_GLOBAL=0 la devuelve.
BANCO_GLOBAL = os.environ.get("AGIX_BANCO_GLOBAL", "1") == "1"
TAU_CORTE = float(os.environ.get("AGIX_TAU", 0.0))   # 0 = sin regla de parada


# Extractor de rasgos reescrito el 21/10/2026 (prueba de extremo a extremo). El de
# antes cortaba por el PRIMER separador y tiraba la cabeza, y la cabeza es justo donde
# va la funcion central: «A web application to book home cleaning services, with ...»
# daba solo los rasgos detras de «with», sin «book home cleaning services»; «graduates
# apply with their CV» daba «their CV». La seleccion cubria los rasgos equivocados.
# verbos que se coordinan compartiendo objeto: «find and book study rooms»
_VERBO_COORD = {"find", "search", "browse", "view", "see", "create", "edit", "publish",
                "book", "order", "track", "manage", "compare", "buy", "sell", "rate",
                "review", "share", "schedule", "list", "upload", "download", "read",
                "write", "post", "pay", "add", "remove", "update", "check", "monitor"}
# verbos en 3.a persona detras de un sujeto: «companies publish», «couriers get»
_VERBO_SUJ = {"publish", "publishes", "apply", "applies", "get", "gets", "order", "orders",
              "pay", "pays", "book", "books", "find", "finds", "run", "runs", "notify",
              "notifies", "receive", "receives", "post", "posts", "manage", "manages",
              "track", "tracks", "see", "sees", "upload", "uploads", "rate", "rates"}
_PRODUCTO = re.compile("(?:app|application|system|platform|portal|tool|software|website|"
                       "site|service|marketplace|dashboard|suite)" + _BB, re.I)

def cabeza_producto(desc):
    """«A mobile app», «Job portal»: el sustantivo de producto sin su complemento."""
    d = (desc or "").strip()
    m = _PRODUCTO.search(d)
    return d[:m.end()].strip() if m else ""

# Extractor v2 (07/11/2026). El v1 se ajusto sobre las 6 descripciones de la prueba de
# extremo a extremo y no generalizo. Sobre las 58 de calibracion salian anclas como
# «manage the owner schedules the walks» o «receive a reminder one day before. Stylists
# see my day's schedule». Causas: no cortaba por oraciones; solo reconocia un punado de
# verbos, asi que no separaba el sujeto; partia por «and» aunque continuara el objeto
# («sees incoming orders and their status» -> «manage my status»); y cuando no entendia
# la clausula le pegaba «manage» igual. Regla nueva: MEJOR NINGUNA ANCLA QUE UNA ROTA.
# Un ancla que falta deja el comportamiento de antes; una rota mete una historia basura.
# Banco de prueba: las 58 descripciones validas de calibracion, traducidas como en
# produccion, mas las 6 del test humano.
_VERBOS_BASE = set("""
accept access add alert allow analyze apply approve arrange ask assemble assign attach attend
build broadcast let make meet play rank score stream win earn trade redeem exchange
pass repeat
bid block book browse buy calculate call cancel change charge chat check choose claim
clean close coach collect comment compare complete configure confirm consult contact
control cook coordinate count create decline deliver deposit describe design dispatch
distribute divide donate download drive edit enrol enroll estimate evaluate explain export
file fill find finish fix follow freeze generate get give grade handle have hire
import inform inspect install invest invite issue join keep learn leave list listen load
log maintain manage mark match measure message monitor notify offer open order organise
organize park pay pick place plan post prepare print process program publish purchase put
quote rate read rebook receive record refill refund register reject remind remove renew
rent reorder repair reply report request reschedule reserve resume return review ride run
save scan schedule search see select sell send serve set share ship show sign sort split
start study submit subscribe suggest support take teach tell test track train transfer
unlock update upgrade upload use validate verify view vote warn wash watch withdraw write
""".split())
# verbos que, sin determinante detras, suelen ser sustantivos: «change notifications»,
# «report cards», «transfers between stores», «study rooms»
_AMBIGUOS = set("""
alert bill block book call change charge check claim comment control count deposit file
list log mark match message order park plan post print process program quote rate record
rent report request return review ride run schedule search set share ship show sign split
start study test track transfer update use view vote watch
""".split())
_IRREGULARES = {"has": "have", "does": "do", "goes": "go"}
# tras un ambiguo, una interrogativa indica verbo: «shows how much I saved»
_INTERROG = {"how", "what", "which", "where", "who", "whether"}
_SER = {"is", "are", "was", "were", "be", "been"}
_DET = {"the", "a", "an", "their", "his", "her", "its", "my", "your", "our", "each",
        "every", "all", "this", "that", "these", "those", "some", "any", "it", "them",
        "you", "me", "us"}
_MODALES = {"can", "could", "must", "should", "may", "might", "will", "would", "shall"}
_PRONOMBRES = {"it", "them", "this", "that", "these", "those", "him", "her"}
_SUBORD = {"before", "after", "when", "once", "until", "while", "because", "if", "whenever"}
# nombres que van en singular pero conjugan en plural: «office staff handle complaints»
_COLECTIVOS_PL = {"staff", "people", "personnel", "police", "crew", "faculty"}
# «alerts you when you overstay the budget» -> «receive an alert when I overstay ...»
_AVISOS = {"alert": ["an", "alert"], "notify": ["a", "notification"],
           "remind": ["a", "reminder"], "warn": ["a", "warning"], "tell": ["a", "message"],
           "inform": ["a", "message"]}
_NOMBRES_AVISO = ("alert", "notification", "reminder", "warning", "message", "notice")


def _lema_verbo(w):
    """forma base si la palabra es un verbo del lexico, o None"""
    w = (w or "").lower().strip(".,;:()")
    if w in _VERBOS_BASE:
        return w
    if w in _IRREGULARES:
        return _IRREGULARES[w]
    cands = []
    if w.endswith("ies"):
        cands.append(w[:-3] + "y")
    if w.endswith("es"):
        cands.append(w[:-2])
    if w.endswith("s"):
        cands.append(w[:-1])
    return next((c for c in cands if c in _VERBOS_BASE), None)


def _plural(w):
    w = (w or "").lower()
    return w.endswith("s") and not w.endswith(("ss", "us", "is"))


def _verbo_inicial(ws):
    """forma base si la clausula EMPIEZA por un verbo, o None si es una frase nominal.
    Un ambiguo sin determinante detras es sustantivo si es tercera persona («transfers
    between stores») o si le sigue un plural («change notifications», «report cards»), y
    verbo si le sigue otra cosa («transfer money», «sign up»). Los verbos en los que el
    extractor v1 ya confiaba se respetan («book rooms», «track shipments»)."""
    if not ws:
        return None
    w0 = ws[0].lower()
    b0 = _lema_verbo(w0)
    if not b0:
        return None
    sig = ws[1].lower() if len(ws) > 1 else ""
    if sig in _SER:
        return None                        # «order is delivered at home»: sujeto pasivo
    if b0 in _AMBIGUOS and sig not in _DET and sig not in _INTERROG:
        fiable = w0 == b0 and (w0 in _VERBO_COORD or w0 in _VERBO_SUJ or w0 == "file")
        if not fiable and (w0 != b0 or _plural(sig) or not sig):
            return None
    return b0


def _separar_sujeto(p):
    """(sujeto, rasgo) con el verbo ya en forma base, o ("", p) si no hay verbo."""
    ws = p.split()
    if ws and ws[0].lower() in _MODALES:          # «... and can cancel them»
        ws = ws[1:]
        p = " ".join(ws)
    if not ws:
        return "", p
    bajo = [w.lower() for w in ws]
    # «for the service office to block spaces when there are events»
    if bajo[0] == "for" and "to" in bajo[1:7]:
        k = bajo.index("to", 1)
        b = _lema_verbo(ws[k + 1]) if k + 1 < len(ws) else None
        if b:
            return " ".join(ws[1:k]), " ".join([b] + ws[k + 2:])
    # verbo al principio: forma base, o tercera persona con el sujeto elidido por la
    # coordinacion («... and shows a QR at the door»)
    b0 = _verbo_inicial(ws)
    if b0:
        return "", " ".join([b0] + ws[1:])
    if _lema_verbo(ws[0]):
        return "", p                       # frase nominal que empieza por un ambiguo
    con_det = bajo[0] in _DET
    for j in range(1, min(6, len(ws))):     # el verbo puede ir al final: «applicants register»
        w = bajo[j]
        if w in _MODALES and j + 1 < len(ws) and _lema_verbo(ws[j + 1]):
            return " ".join(ws[:j]), " ".join([_lema_verbo(ws[j + 1])] + ws[j + 2:])
        b = _lema_verbo(w)
        if not b:
            continue
        sig = bajo[j + 1] if j + 1 < len(ws) else ""
        if sig in _SER:
            continue                       # «the order is delivered»: pasiva
        if w != b and b in _AMBIGUOS and sig not in _DET and not con_det:
            continue                       # «sales reports of ...»: sustantivo
        # concordancia: un verbo en forma base pide sujeto plural («managers schedule
        # shifts»); con sujeto singular y sin determinante es un nombre compuesto
        # («inventory control for ...»). Con determinante se acepta igual, porque el
        # traductor se come a veces la -s: «the customer schedule the service».
        suj = bajo[:j]
        plural = (_plural(suj[-1]) or suj[-1] in _COLECTIVOS_PL or "and" in suj
                  or suj[-1] in ("i", "we", "you", "they"))
        if w == b and not plural and not con_det:
            continue
        return " ".join(ws[:j]), " ".join([b] + ws[j + 1:])
    return "", p


def _tiene_verbo(p):
    suj, r = _separar_sujeto(p)
    return bool(suj) or r != p or _verbo_inicial(r.split()) is not None


def _verbo_tras_nombre(p):
    """verbo en la 2.a o 3.a palabra: «groom put together ...» continua un sujeto"""
    return any(_lema_verbo(w) for w in p.split()[1:3])


def rasgos_de(desc):
    return [r for _, r in rasgos_con_sujeto(desc)]


def rasgos_con_sujeto(desc):
    """(sujeto, rasgo) por cada funcion que la descripcion enumera. El sujeto es lo
    que va delante del verbo («companies» en «companies publish offers»), o ""; el rasgo
    empieza por el verbo en forma base si lo hay."""
    d = re.sub("[(][^)]*[)]", " ", (desc or "")).strip().rstrip(".")
    # primero por oraciones: «... one day before. Stylists see my day's schedule» son dos
    oraciones = [o.strip().rstrip(".") for o in re.split("[.] +(?=[A-Z])", d) if o.strip()]
    trozos, oracion_de = [], []
    for i_o, o in enumerate(oraciones):
        for x in re.split(",|;|:", o):
            if x.strip():
                trozos.append(x.strip())
                oracion_de.append(i_o)
    if not trozos:
        return [("", d)]
    # lo que va antes de los primeros dos puntos es el titulo del producto
    o0 = oraciones[0]
    titulo = (":" in o0 and o0.index(":") < min(o0.index(x) if x in o0 else len(o0)
                                               for x in (",", ";")))
    # la cabeza «A system for a small clinic to manage records»: se queda lo que hace
    cab = trozos[0]
    m = re.search(_BB + "(to|where|so that|that lets|that|which)" + _BB, cab, re.I)
    if m:
        trozos[0] = cab[m.end():].strip()
    elif _PRODUCTO.search(cab):
        partes = re.split(_BB + "with" + _BB, cab, maxsplit=1, flags=re.I)
        trozos[0] = partes[1].strip() if len(partes) > 1 else ""
    elif not _tiene_verbo(cab) and (titulo or re.match("(a|an|the) ", cab, re.I)
                                    or re.search(_BB + "for (a|an|the)" + _BB, cab, re.I)):
        # «An online shop for handmade jewelry», «Wholesale orders for an industrial
        # bakery»: el producto, no una funcion
        trozos[0] = ""
    out = []
    # Sujeto elidido por la coordinacion: en «managers schedule shifts and review employee
    # contracts», la segunda clausula no dice quien, y el rol se elegia por parecido con el
    # objeto («As an employee, I want to review employee contracts»). Dentro de la MISMA
    # oracion, una clausula que empieza por verbo hereda el ultimo sujeto visto.
    suj_previo, oracion_previa = "", -1
    for t, i_o in zip(trozos, oracion_de):
        if i_o != oracion_previa:
            suj_previo, oracion_previa = "", i_o
        # «covering timetables and ...» = «que incluye horarios y ...»
        t = re.sub("^(?:and|with|where|including|offering|plus|covering|featuring|"
                   "supporting|providing|to|that)" + _BB, "", t.strip(), flags=re.I).strip()
        if not t:
            continue
        piezas, act = [], ""
        for w in re.split(_BB + "and" + _BB, t):
            w = w.strip()
            if act and act.split()[-1].lower() in _VERBO_COORD:
                act = act + " and " + w        # «find and book study rooms»
            elif act and _tiene_verbo(act) and not _tiene_verbo(w):
                act = act + " and " + w        # «sees incoming orders and their status»
            elif (act and len(act.split()) <= 3 and not _tiene_verbo(act)
                  and _verbo_tras_nombre(w)):
                act = act + " and " + w        # «the bride and groom put together ...»
            else:
                if act:
                    piezas.append(act)
                act = w
        if act:
            piezas.append(act)
        for p_ in piezas:
            original = p_
            suj, p_ = _separar_sujeto(p_)
            if suj:
                suj_previo = suj
            elif suj_previo and p_ != original or (suj_previo and _verbo_inicial(p_.split())):
                suj = suj_previo           # clausula que empieza por verbo: hereda
            if len(p_) >= 6:
                out.append((suj, p_))
    return out[:8] or [("", d)]


# ------------------------------------- anclas: una historia por funcion descrita
# Prueba de extremo a extremo (21/10/2026): en salas de estudio el banco traia 1
# candidata de reservar sala entre 28, y en la ferreteria 0 de transferencias. No es la
# seleccion: lo que no se genera no se puede elegir. Y no hay forma de pedirlo por la
# entrada: con «A mobile app: find and book study rooms» el modelo sigue escribiendo
# lecciones y calificaciones, porque la ficha de funcion son seis verbos sin objeto y el
# decodificador copia poco objeto de la entrada (lo mismo que la plantilla CRUD).
# Lo que si obedece es un PREFIJO forzado en el decodificador: se le da
# «As a student, I want to find and book study rooms on campus in real time, so that»
# y escribe el beneficio. La capacidad queda trazada a la descripcion, que es lo que
# pide un requisito, y el beneficio y las tareas siguen saliendo de los modelos.
ANCLAR = os.environ.get("AGIX_ANCLAR", "1") == "1"
N_BEN_ANCLA = int(os.environ.get("AGIX_N_BEN_ANCLA", 6))
_LEMA = {"publishes": "publish", "applies": "apply", "gets": "get", "orders": "order",
         "pays": "pay", "books": "book", "finds": "find", "runs": "run",
         "notifies": "notify", "receives": "receive", "posts": "post",
         "manages": "manage", "tracks": "track", "sees": "see", "uploads": "upload",
         "rates": "rate"}
_POSESIVO = {"their": "my", "his": "my", "her": "my", "its": "my", "your": "my"}
# sujetos colectivos que SBERT no casa con su rol: «companies» quedaba mas cerca de
# «graduate» que de «recruiter» y salia «Como graduado, quiero publicar ofertas»
_SUJETO_ROL = {"company": ("recruiter", "employer", "hiring manager", "company"),
               "employer": ("recruiter", "employer", "hiring manager"),
               "business": ("business owner", "owner", "seller", "vendor"),
               "shop": ("store manager", "seller", "owner"),
               "store": ("store manager", "seller", "owner")}


def capacidad_de_rasgo(rasgo, sujeto=""):
    """El rasgo como capacidad: «publish offers» tal cual; un sustantivo lleva verbo
    ligero («low stock alerts» -> «receive low stock alerts»). Devuelve "" si la clausula
    no se puede convertir en una capacidad limpia: mejor ninguna ancla que una rota.
    Si el rasgo vino con sujeto, su primera palabra ES un verbo y no se vuelve a dudar."""
    ws = rasgo.split()
    if not ws:
        return ""
    w0 = ws[0].lower()
    b = (_lema_verbo(w0) if sujeto else None) or _verbo_inicial(ws) or _LEMA.get(w0)
    if b and len(ws) > 1 and ws[1].lower() in _SER:
        return ""                      # pasiva: «order is delivered at home»
    if b:
        ws = [b] + ws[1:]
        if b in _AVISOS and len(ws) > 1 and ws[1].lower() in ("you", "me", "us"):
            ws = ["receive"] + _AVISOS[b] + ws[2:]
        elif len(ws) > 1 and ws[1].lower() in _PRONOMBRES:
            return ""                  # «validate it»: el objeto quedo en otra clausula
        # la subordinada no es la funcion («approve the budget ... before the work
        # begins»), salvo en un aviso, donde el «cuando» ES lo que se pide
        corte = next((i for i, w in enumerate(ws)
                      if 2 <= i < len(ws) - 1 and w.lower() in _SUBORD), None)
        if corte and not any(x.lower().startswith(_NOMBRES_AVISO) for x in ws[:corte]):
            ws = ws[:corte]
        # segundo verbo coordinado en tercera persona: «close and divides the accounts»
        for i in range(1, len(ws)):
            if ws[i - 1].lower() in ("and", "or"):
                lv = _lema_verbo(ws[i])
                if lv and lv != ws[i].lower() and lv not in _AMBIGUOS:
                    ws[i] = lv
    else:
        if any(w.lower() in _SER for w in ws):
            return ""                  # pasiva o «there is ...»: no es una funcion
        con_sub = any(w.lower() in _SUBORD for w in ws)
        if len(ws) > (10 if con_sub else 8):
            return ""                  # una frase nominal larga es una oracion mal partida
        bajo = rasgo.lower()
        if re.search("alert|notification|reminder", bajo):
            ws = ["receive"] + ws
        elif re.search("report|history|statistic|dashboard|summary|analytics", bajo):
            ws = ["view"] + ws
        else:
            ws = ["manage"] + ws
    if len(ws) > 12:
        return ""
    # «apply with their CV» -> «my CV»: el posesivo es del actor. Pero en «see incoming
    # orders and their status» es de los pedidos: con un plural antes se deja como esta.
    # «you» en la subordinada pasa a «I»: «when you overstay» -> «when I overstay».
    salida = []
    for i, w in enumerate(ws):
        plural_antes = any(_plural(x) and x.lower() not in _DET
                           and x.lower() not in _VERBOS_BASE for x in ws[1:i])
        if w.lower() == "you" and i > 0:
            salida.append("I")
        else:
            salida.append(w if plural_antes else _POSESIVO.get(w.lower(), w))
    return " ".join(salida)


def seleccion_por_rasgos(sb, cands, desc, k, extra=None, lam=0.15, mu=0.15,
                         roles=None, bens=None, nu=0.10, pena=None, clases=None,
                         mu_clase=0.12, caps=None, nu_cap=0.0, piso_cap=0.75,
                         cupo_cap=0):
    """F(S) = sum_rasgo max_{s in S} sim(rasgo, s), voraz. Es la localizacion de
    instalaciones sobre los RASGOS de la descripcion, no sobre el banco de
    candidatas: cubrir el banco es cubrir la distribucion del propio modelo, que
    es el objetivo equivocado. Aqui cada rasgo enumerado tira de una historia."""
    rasgos = rasgos_de(desc)
    Er = sb.encode(rasgos, normalize_embeddings=True, show_progress_bar=False)
    Ec = sb.encode(cands, normalize_embeddings=True, show_progress_bar=False)
    M = Er @ Ec.T                                  # rasgos x candidatas
    cubierto = np.zeros(len(rasgos), dtype=np.float32)
    elegidos, libre = [], np.ones(len(cands), bool)
    vistos_rol, vistos_ben, vistos_clase = set(), set(), set()
    # Capacidad repetida POR ACTOR. El termino `nu` de arriba solo ve el beneficio
    # literal, y el deduplicador de historias compara la historia COMPLETA, donde un
    # beneficio distinto diluye el parecido por debajo del umbral. Resultado leido en un
    # colegio: tres historias de «padre» que piden lo mismo (notificaciones de las
    # proximas lecciones) y solo se diferencian en el «para que». Aqui se mide el
    # parecido de la CAPACIDAD, y solo contra las ya elegidas DEL MISMO ACTOR, porque
    # que el padre y el estudiante reciban notificaciones si son historias distintas.
    Scap, repes = None, 0
    if caps is not None and nu_cap:
        Ecp = sb.encode([c or "" for c in caps], normalize_embeddings=True,
                        show_progress_bar=False)
        Scap = (Ecp @ Ecp.T).astype(np.float32)
    for _ in range(min(k, len(cands))):
        gan = np.maximum(cubierto[:, None], M).sum(0) - cubierto.sum()
        # desempate por relevancia al conjunto de la descripcion, para que cuando
        # los rasgos ya esten cubiertos siga eligiendo lo pertinente
        gan = gan + 0.05 * M.mean(0)
        if roles is not None and mu:
            # Cobertura de ROLES, tercer termino. Sin esto el reordenador de
            # beneficios concentra los actores: una vez cubiertos los rasgos, los
            # huecos restantes se los lleva quien puntue mejor el beneficio, y eso
            # es sistematicamente el mismo rol (los beneficios de "mensajero" son
            # formulaicos y puntuan alto). Medido en una farmacia: 7 de 12
            # historias del mismo actor. Un rol nuevo vale mu; cuando ya estan
            # todos cubiertos el termino deja de pesar, asi que no es una cuota.
            gan = gan + mu * np.array([0.0 if r in vistos_rol else 1.0 for r in roles])
        if clases is not None and mu_clase:
            # Cobertura de CLASES de funcion, hermano del termino de roles. Sin esto la
            # ficha solo diversifica el banco: la seleccion por cobertura de rasgos no
            # sabe de clases y puede volver a elegir doce historias de tipo view.
            gan = gan + mu_clase * np.array(
                [0.0 if c in vistos_clase else 1.0 for c in clases])
        if bens is not None and nu:
            # Beneficio LITERALMENTE repetido. Medido sobre 6 backlogs: 63 de 68
            # clausulas "para que" son distintas, asi que esto no es un problema
            # general; pero cuando se repite lo hace en bloque (4 de 12 historias
            # de un taller mecanico acababan en "para poder planificar mi horario")
            # y ahi el backlog se lee como una plantilla. Solo castiga la
            # coincidencia exacta, no el parecido, para no volver a pagar
            # atribucion por quitar plantilla.
            gan = gan - nu * np.array([1.0 if b_ and b_ in vistos_ben else 0.0
                                       for b_ in bens])
        if Scap is not None and elegidos and repes >= cupo_cap:
            # rampa desde el piso hasta 1, no binaria: asi el castigo es proporcional
            # al parecido y no depende de acertar un umbral exacto
            pc = np.zeros(len(cands), dtype=np.float32)
            for i_ in range(len(cands)):
                if not libre[i_]:
                    continue
                mismo = [j_ for j_ in elegidos
                         if roles is None or roles[j_] == roles[i_]]
                if not mismo:
                    continue
                m_ = float(Scap[i_, mismo].max())
                if m_ > piso_cap:
                    pc[i_] = nu_cap * (m_ - piso_cap) / (1.0 - piso_cap)
            gan = gan - pc
        if pena is not None:
            gan = gan - pena
        if extra is not None:
            # Coherencia del beneficio como BONIFICACION, no como filtro previo.
            # En cascada (recortar el banco a 2k por coherencia y luego cubrir) se
            # perdian las candidatas que eran las unicas que cubrian un rasgo
            # minoritario: los backlogs completos caian del 66,7 % al 53,3 %. Aqui
            # una candidata que aporta un rasgo nuevo gana aunque su beneficio sea
            # mediocre, y entre candidatas redundantes manda el beneficio.
            gan = gan + lam * extra
        gan[~libre] = -np.inf
        j = int(np.argmax(gan))
        if Scap is not None and elegidos:
            mismo = [j_ for j_ in elegidos if roles is None or roles[j_] == roles[j]]
            if mismo and float(Scap[j, mismo].max()) > piso_cap:
                repes += 1          # se gasta el cupo, aunque no se haya castigado
        elegidos.append(j)
        libre[j] = False
        if roles is not None:
            vistos_rol.add(roles[j])
        if bens is not None and bens[j]:
            vistos_ben.add(bens[j])
        if clases is not None:
            vistos_clase.add(clases[j])
        cubierto = np.maximum(cubierto, M[:, j])
    return elegidos


def seleccion_submodular(S, k, rel=None, lam=0.0, tau=0.0):
    """S: matriz de similitud VxV (simetrica). Devuelve los indices elegidos.
    rel: relevancia de cada candidata a la consulta, opcional. tau: corta cuando
    la ganancia marginal cae por debajo de tau veces la primera."""
    n = S.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    cubierto = np.zeros(n, dtype=np.float32)
    elegidos, libre = [], np.ones(n, bool)
    primera = None
    for _ in range(k):
        # ganancia de anadir j = cuanto sube sum_v max(cubierto_v, S[v, j])
        nuevo = np.maximum(cubierto[:, None], S)          # V x V
        gan = nuevo.sum(0) - cubierto.sum()
        if rel is not None and lam:
            gan = gan + lam * n * rel
        gan[~libre] = -np.inf
        j = int(np.argmax(gan))
        if not np.isfinite(gan[j]):
            break
        if primera is None:
            primera = gan[j]
        elif tau and primera > 0 and gan[j] < tau * primera:
            break                                          # el contenido decide el tamano
        elegidos.append(j)
        libre[j] = False
        cubierto = np.maximum(cubierto, S[:, j])
    return elegidos


# Taxonomia de sectores que usa el planificador de roles. v4 (07/11/2026) son los 40
# sectores de v3 mas 11 nuevos (src/93_dominios_v4.py); queda en v3 por omision hasta
# validarla con el conjunto de calibracion, porque cambia el plan de roles de cualquier
# descripcion cuyo sector mas parecido pase a ser uno de los nuevos.
TAXONOMIA = os.environ.get("AGIX_TAXONOMIA", "descripciones_v3.json")


class PlanificadorRoles:
    def __init__(self, sb):
        p = f"{BASE}/data/synth/{TAXONOMIA}"
        self.ok = os.path.exists(p)
        if not self.ok:
            return
        D = json.load(open(p, encoding="utf-8"))
        self.textos = [x["descripcion"] for x in D]
        self.roles = [x["roles"] for x in D]
        self.E = sb.encode(self.textos, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=128)
        self.sb = sb

    def __call__(self, descripcion, k=4, previos=()):
        previos = [r for r in (previos or ()) if r]
        if not self.ok:
            base = actores_nombrados(descripcion, k) if EXTRAER_ROLES else []
            return (previos + [r for r in base if r not in previos])[:k]
        e = self.sb.encode([descripcion], normalize_embeddings=True,
                           show_progress_bar=False)[0]
        sim = self.E @ e
        orden = np.argsort(-sim)
        if float(sim[orden[0]]) < UMBRAL_SECTOR:
            # dominio desconocido: no se fuerza un plantel ajeno, se leen los
            # actores que la descripcion nombra. Hacen falta DOS: con uno solo el
            # backlog entero sale del mismo actor, y si ese actor esta mal leido es
            # peor que no forzar nada. Medido en el dominio de integracion continua:
            # la descripcion dice "build pipelines on agents" y el extractor saca un
            # unico "agent" (que ahi es una MAQUINA), asi que salian 12 historias
            # seguidas de "Como agente". Con menos de dos se deja al modelo libre.
            if len(previos) >= 2:
                ext = list(previos)
            else:
                ext = actores_nombrados(descripcion, k) if EXTRAER_ROLES else []
                ext = [r for r in ext if not rol_infraestructura(descripcion, r)]
                ext = previos + [r for r in ext if r not in previos]
            if len(ext) < MIN_ACTORES_SIN_PLANTEL:      # paso 4b, en tres pasos
                for r in actores_colectivos(descripcion, k):
                    if r not in ext:
                        ext.append(r)
            if len(ext) < MIN_ACTORES_SIN_PLANTEL:
                for r in ROLES_GENERICOS:
                    if r not in ext and len(ext) < MIN_ACTORES_SIN_PLANTEL + 1:
                        ext.append(r)
            return ext[:k]                              # ya nunca queda vacio
        plantel = []
        for i in orden[:3]:                       # los 3 sectores mas parecidos
            for r in self.roles[int(i)]:
                if r not in plantel:
                    plantel.append(r)
        self.ultimo_sector = float(sim[orden[0]])
        if self.ultimo_sector < UMBRAL_PLANTEL_LIBRE:
            plantel = [r for r in plantel if rol_respaldado(descripcion, r)]
            # el plantel ya no garantiza dos actores: se completa como sin plantel
            hueco = list(previos) + [r for r in actores_nombrados(descripcion, k)
                                     if r not in previos] if EXTRAER_ROLES else list(previos)
            hueco = hueco + [r for r in plantel if r not in hueco]
            # «reservar servicios de limpieza ... pagos en la app» no nombra a nadie, pero
            # quien reserva y paga es un cliente. Solo si ningun actor de consumo esta ya
            if (_hay_transaccion(descripcion)
                    and not any(r in _CONSUMO for r in hueco)):
                hueco.insert(0, "customer")
            if len(hueco) < MIN_ACTORES_SIN_PLANTEL:
                for r in actores_colectivos(descripcion, k):
                    if r not in hueco:
                        hueco.append(r)
            if len(hueco) < MIN_ACTORES_SIN_PLANTEL:
                for r in ROLES_GENERICOS:
                    if r not in hueco and len(hueco) < MIN_ACTORES_SIN_PLANTEL:
                        hueco.append(r)
            return hueco[:k]
        if MODO_ROLES.startswith("descripcion") and EXTRAER_ROLES:
            # Si la FUENTE en espanol ya nombra dos actores o mas, se le cree y no se
            # consulta el extractor sobre la traduccion: su entrada es una version con
            # perdidas de la misma frase, asi que aporta ruido antes que cobertura
            # («apoderados» -> «agents» metia un actor que no existe en el colegio).
            # Con menos de dos, el extractor ingles sigue cubriendo el hueco.
            if len(previos) >= 2:
                ext = list(previos)
            else:
                ext = actores_nombrados(descripcion, k)
                ext = previos + [r for r in ext if r not in previos]
            if len(ext) >= MIN_ACTORES_CON_PLANTEL:   # la descripcion nombra a su gente
                # descripcion_solo: NADA de relleno. Rellenar con el plantel del
                # sector reintroduce actores ajenos justo donde la descripcion ya
                # dijo quien es su gente (supermercado: las personas escribieron
                # customer y employee, y el relleno metia shopper y picker).
                if MODO_ROLES == "descripcion_solo":
                    return ext[:k]
                return (ext + [r for r in plantel if r not in ext])[:k]
        # si la descripcion nombra un actor, ese manda
        d = descripcion.lower()
        nombrados = [r for r in plantel if r in d or (r + "s") in d]
        resto = [r for r in plantel if r not in nombrados]
        return (nombrados + resto)[:k]


def _marian(ident, carpeta):
    """Los dos traductores, desde una carpeta del paquete en vez del cache de Hugging
    Face. Con el cache, estando online la libreria vuelve a resolver `main` contra el
    hub y se redescarga la otra variante de los pesos: el paquete engordaba 600 MB y
    dejaba de ser offline sin avisar. Si la carpeta no esta, se cae al identificador
    de siempre."""
    local = f"{BASE}/models/{carpeta}"
    return local if os.path.isdir(local) else ident

class Pipeline:
    def __init__(self, traducir=False, conjunto=CONJUNTO):
        self.conjunto = conjunto
        t0 = time.time()
        self.sb = SentenceTransformer(SBERT, device=DISPOSITIVO)
        self.sb.max_seq_length = 128
        self.tok_a = AutoTokenizer.from_pretrained(M_ARRIBA)
        self.m_a = MT5ForConditionalGeneration.from_pretrained(M_ARRIBA, dtype=torch.float32).to(DISPOSITIVO).eval()
        self.tok_b = AutoTokenizer.from_pretrained(M_ABAJO)
        self.m_b = MT5ForConditionalGeneration.from_pretrained(M_ABAJO, dtype=torch.float32).to(DISPOSITIVO).eval()
        self.m_a.config.use_cache = self.m_b.config.use_cache = True
        # La ficha se decide POR MODELO, no por variable de entorno: solo tiene
        # sentido con un modelo entrenado con ella (los conjuntos v3r). Si se
        # decidiera por entorno, en una comparacion pareada la pasada del modelo
        # viejo tambien recibiria la ficha, que para el es ruido, y el resultado
        # estaria sesgado a favor del nuevo. AGIX_FICHA_ROL sigue valiendo para
        # forzarla a mano en experimentos.
        # Las dos fichas se deciden por el NOMBRE del modelo. Todo conjunto entrenado
        # con ellas tiene que estar en esta lista: si se despliega un modelo nuevo y se
        # olvida, el producto corre SIN fichas y se pierden los pasos 2 y 4 sin avisar.
        CON_FICHA = ("v3r", "v8ff", "v9bt", "v9df")
        usar = (any(x in M_ARRIBA for x in CON_FICHA)
                or os.environ.get("AGIX_FICHA_ROL") == "1"
                or REPARAR_ROL)      # la reparacion tambien necesita el plantel
        # la ficha de funcion solo tiene sentido con un modelo entrenado con ella
        self.usa_funcion = (MODO_FUNCION == "1" or
                            (MODO_FUNCION == "auto"
                             and any(x in M_ARRIBA for x in ("v8ff", "v9bt", "v9df"))))
        self.planificador = PlanificadorRoles(self.sb) if usar else None
        # reordenador de beneficios: se carga si existe y no se apaga a mano
        self.reordenador = None
        _rp = f"{BASE}/models/reordenador_beneficio.joblib"
        if os.environ.get("AGIX_REORDENAR", "1") == "1" and os.path.exists(_rp):
            import joblib
            self.reordenador = joblib.load(_rp)["modelo"]
        if self.planificador is not None and not self.planificador.ok:
            print(f"  AVISO: no hay {TAXONOMIA}, se desactiva la ficha de rol")
            self.planificador = None
        # Tercer generador del conjunto. Elegido midiendo el punto de operacion
        # (logs/66_ratio.log): con este el numero de tareas sube un 46 % y la
        # concentracion de plantilla baja a 34,2 %, la mas baja conseguida, que
        # es lo que importa porque el texto de la tarea es la entrada de la
        # cadena de estimacion. Alternativas medidas, por si se prefiere otro
        # equilibrio: ..._hist_agix_v1_v3_agix_v1 da mejor atribucion (71,5 %) y
        # anclaje (88,3 %) con mas plantilla (52,1 %); ..._hist_30sec da mas
        # tareas (667) y lo peor en todo lo demas. AGIX_ABAJO3="" lo desactiva.
        self.m_b3 = None
        if M_ABAJO3:
            # tercer generador del conjunto: sube el numero de tareas, que es el
            # defecto de primer orden (generamos ~la mitad de las de referencia)
            self.m_b3 = MT5ForConditionalGeneration.from_pretrained(
                M_ABAJO3,
                dtype=torch.float32).to(DISPOSITIVO).eval()
            self.m_b3.config.use_cache = True
        self.m_b2 = None
        if conjunto != "no":
            self.m_b2 = MT5ForConditionalGeneration.from_pretrained(
                M_ABAJO2, dtype=torch.float32).to(DISPOSITIVO).eval()
            self.m_b2.config.use_cache = True

        # --- retrieval pools, built from TRAIN projects only
        A = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_train.parquet",
                            columns=["parent_uid", "entrada", "objetivo", "proj"])
        A = A.groupby("parent_uid").agg(entrada=("entrada", "first"), proj=("proj", "first"),
                                        hijos=("objetivo", list)).reset_index()
        B = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_train.parquet",
                            columns=["parent_uid", "entrada_en", "tareas", "proj", "mitad", "estilo"])
        B = B[(B.mitad == "tasks") & (B.estilo == "jira")].drop_duplicates("parent_uid")
        self.A, self.B = A, B.reset_index(drop=True)
        if os.path.exists(CACHE_EMB):
            z = np.load(CACHE_EMB)
            self.EA = torch.tensor(z["A"]).to(DISPOSITIVO)
            self.EB = torch.tensor(z["B"]).to(DISPOSITIVO)
        else:
            self.EA = self.sb.encode(A.entrada.tolist(), batch_size=256, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False)
            self.EB = self.sb.encode(self.B.entrada_en.tolist(), batch_size=256, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False)
            np.savez(CACHE_EMB, A=self.EA.cpu().numpy(), B=self.EB.cpu().numpy())
        self.trad_es_en = None
        if TRADUCIR_ENTRADA:
            from transformers import MarianMTModel, MarianTokenizer
            self.tok_ee = MarianTokenizer.from_pretrained(_marian("Helsinki-NLP/opus-mt-es-en", "marian_es_en"),
                                                          cache_dir=f"{BASE}/models/hf_cache")
            self.trad_es_en = MarianMTModel.from_pretrained(
                _marian("Helsinki-NLP/opus-mt-es-en", "marian_es_en"), cache_dir=f"{BASE}/models/hf_cache",
                dtype=torch.float16).to(DISPOSITIVO).eval()
        self.trad = None
        if traducir:
            from transformers import MarianMTModel, MarianTokenizer
            # same local cache as the other translator, so the whole demo travels in one folder
            self.tok_es = MarianTokenizer.from_pretrained(_marian("Helsinki-NLP/opus-mt-en-es", "marian_en_es"),
                                                          cache_dir=f"{BASE}/models/hf_cache")
            self.trad = MarianMTModel.from_pretrained(
                _marian("Helsinki-NLP/opus-mt-en-es", "marian_en_es"), cache_dir=f"{BASE}/models/hf_cache",
                dtype=torch.float16 if DISPOSITIVO == "cuda" else torch.float32).to(DISPOSITIVO).eval()
        self.carga_s = round(time.time() - t0, 1)

    # ------------------------------------------------------------- helpers
    def _pena_forma(self, triples):
        """Castigo por la FORMA de la historia, no por su contenido.

        Dos defectos que se ven al leer y que ninguna metrica de conjunto detecta:

        - la historia se muerde la cola: "As an account holder, I want to be able to
          freeze my card, so that I can freeze immediately". El beneficio repite la
          capacidad en lugar de decir para que sirve.
        - la capacidad divaga: "I want to verify the identity checks when a freeze
          can be processed before a risk if the user has account balanced their
          identity review". Son subordinadas encadenadas; se mide por numero de
          palabras, que es lo unico robusto sin analizador sintactico.

        Para la circularidad NO sirve el coseno del SBERT: medido sobre 24 historias
        generadas, la circular saca 0,523 y una legitima ("to be notified of upcoming
        appointments" / "I can plan my schedule") saca 0,560. El SBERT propio mide
        TEMA, y un beneficio legitimo esta tematicamente cerca de su capacidad por
        definicion. La senal buena es lexica: que el beneficio repita el VERBO de la
        capacidad (saltando "be able to"), o la mitad de sus palabras de contenido.
        Esa regla marca 1 de las 24 y es justo la circular.

        Devuelve un vector que la seleccion resta. Es castigo, no filtro: una
        candidata con la unica cobertura de un rasgo sigue pudiendo ganar.
        """
        if not triples or not any(triples):
            return None
        caps = [(t[1] if t else "") for t in triples]
        circ = np.array([self._circular(t[1], t[2]) if t else 0.0 for t in triples],
                        dtype=np.float32)
        largo = np.array([1.0 if len(c.split()) > TOPE_PALABRAS_CAP else 0.0
                          for c in caps], dtype=np.float32)
        auto = np.array([self._autorreferente(t) if t else 0.0 for t in triples],
                        dtype=np.float32)
        eco = np.array([self._eco_capacidad(c) for c in caps], dtype=np.float32)
        rolfun = np.array([self._rol_funcion(t) if t else 0.0 for t in triples],
                          dtype=np.float32)
        vacio = np.array([1.0 if t and beneficio_vacio(t[2]) else 0.0 for t in triples],
                         dtype=np.float32)
        return (NU_CIRCULAR * circ + NU_LARGO * largo + NU_AUTO * auto + NU_ECO * eco
                + NU_ROLFUN * rolfun + NU_VACIO * vacio)

    def _rol_funcion(self, t):
        """el beneficio presupone que el actor trabaja en la organizacion, y el actor es
        un consumidor: "As a guest, ... so that I can plan my workload". Ver el comentario
        de _RE_TRABAJO_AJENO para la falsacion contra los dos corpus de referencia."""
        if familia_rol_sim(t[0], self.sb) != "consume":
            return 0.0
        return 1.0 if _RE_TRABAJO_AJENO.search(t[2] or "") else 0.0

    @staticmethod
    def _autorreferente(t):
        """el actor se refiere a si mismo: "As a student, I want to view the schedule
        for a student" o "...the status of my student's absences". Un estudiante no
        tiene "mi estudiante". Medido sobre 312 historias generadas: pasa en el 1,9 %
        y las tres que marca son las tres que estan mal, sin falsos positivos. Casi
        todas caen en el mismo dominio (colegio), donde los actores comparten nucleo
        (estudiante, profesor, apoderado)."""
        nucleo = (t[0] or "").strip().lower().split()
        if not nucleo:
            return 0.0
        n = re.sub(r"[^a-z]", "", nucleo[-1])
        if len(n) < 4:
            return 0.0
        resto = f"{t[1]} {t[2]}".lower()
        pat = (rf"\b(?:my|a|an|the|another|all|every|other)\s+(?:\w+\s+){{0,2}}"
               rf"{re.escape(n)}s?\b")
        return 1.0 if re.search(pat, resto) else 0.0

    @staticmethod
    def _eco_capacidad(cap):
        """la misma palabra de contenido dos veces dentro de la capacidad: "view the
        availability of my loved one's availability", "view the lesson status of my
        lessons", "view customer feedback on the customer's history". El filtro
        degenerada_texto no lo ve porque exige tres repeticiones exactas y aqui son
        dos, y una en plural. Medido sobre 372 historias generadas: 3,5 %, y los
        cuatro casos distintos que marca estan los cuatro mal."""
        pal = [singular(w) for w in re.findall(r"[a-z]+", (cap or "").lower())
               if w not in _FUNCION and len(w) > 3]
        return 1.0 if len(pal) != len(set(pal)) else 0.0

    @staticmethod
    def _circular(cap, ben):
        c = re.sub(r"^\s*(to\s+)?(be\s+able\s+to\s+|have\s+the\s+ability\s+to\s+|be\s+)?",
                   "", (cap or "").strip().lower())
        pal = c.split()
        if pal and pal[0] in set(re.findall(r"[a-z]+", (ben or "").lower())):
            return 1.0
        cw = set(re.findall(r"[a-z]{4,}", c)) - _FUNCION
        bw = set(re.findall(r"[a-z]{4,}", (ben or "").lower())) - _FUNCION
        return 1.0 if cw and len(cw & bw) / len(cw) >= 0.5 else 0.0

    @torch.no_grad()
    def _reparar_verbo_en_tareas(self, historias):
        """El imperativo ingles leido como sustantivo, que es el fallo de traduccion
        que mas se nota porque cae en las dos plantillas de tarea mas frecuentes:

          Display order status on the kitchen dashboard
            -> Estado del pedido de visualizacion en el tablero de cocina
          Store booking counts in the database
            -> Cuenta de reservas de tiendas en la base de datos   (store -> tienda)
          Support special requests -> Solicitudes especiales de apoyo

        La tabla de verbos y la deverbal no llegan aqui: el verbo no esta al
        principio, esta disuelto en la frase. La solucion es forzar la lectura
        verbal en el ORIGEN: se vuelve a traducir la tarea como You must <tarea>,
        que el traductor rinde como Debe <infinitivo> ..., y se quita el Debe.
        Probado sobre los seis casos que fallaban: los arregla los seis. Prefijar
        To no sirve, ya se midio: unas veces arregla y otras mete un Para delante.

        Solo se re-traducen las tareas cuya traduccion NO empieza por infinitivo,
        que son alrededor del 5 %. Si la segunda pasada tampoco da un infinitivo,
        se deja la primera: nunca queda peor.
        """
        if self.trad is None:
            return 0
        pend = []
        for h in historias:
            for j, (en, es) in enumerate(zip(h["tareas"], h.get("tareas_es") or [])):
                w = (es.split() or [''])[0].lower()
                if w.endswith(("ar", "er", "ir")) or not en or not en[0].isalpha():
                    continue
                pend.append((h, j, en))
        if not pend:
            return 0
        fuentes = [f"You must {en[0].lower()}{en[1:]}" for _, _, en in pend]
        x = self.tok_es(fuentes, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(DISPOSITIVO)
        with torch.no_grad():
            y = self.trad.generate(**x, num_beams=1, max_new_tokens=96)
        arregladas = 0
        salida = self.tok_es.batch_decode(y, skip_special_tokens=True)
        for (h, j, _), t_ in zip(pend, salida):
            t_ = re.sub(r"^\s*(?:usted\s+)?(?:debe(?:s|ria|rias)?|tiene\s+que|hay\s+que|necesita(?:s|mos)?|tenemos\s+que)\s+", "", t_, flags=re.I).strip()
            if not t_:
                continue
            t_ = t_[0].upper() + t_[1:]
            if (t_.split() or [""])[0].lower().endswith(("ar", "er", "ir")):
                h["tareas_es"][j] = glosario(t_)
                arregladas += 1
        return arregladas
    @torch.no_grad()
    def _reparar_rol_en_tareas(self, historias):
        """El actor se nombra en la tarea inglesa y se pierde al traducir.

        Display the delivery time on the courier's dashboard sale como Mostrar el
        tiempo de entrega en el panel de correo: el traductor rinde courier como
        correo dentro de la tarea aunque la historia diga mensajero, y la tarea
        parece hablar de otra cosa. Medido sobre 1.421 tareas generadas: pasa en el
        6,4 %, con casos como cleaner -> panel de limpieza, o host -> al host.

        Se vuelven a traducir SOLO esas tareas, con el nombre del actor sustituido
        por una ficha que se restaura con el nombre en espanol que ya usa la
        historia. Es la misma maquinaria de proteger/restaurar de los terminos
        tecnicos, aplicada al actor.
        """
        if self.trad is None:
            return 0
        pend = []
        for h in historias:
            rol_en, rol_es = rol_de(h["historia"]), rol_es_de(h.get("historia_es"))
            if not rol_en or not rol_es or not h.get("tareas_es"):
                continue
            nen, nes = rol_en.split()[-1], rol_es.split()[-1]
            for j, (en, es) in enumerate(zip(h["tareas"], h["tareas_es"])):
                if (re.search(rf"\b{re.escape(nen)}s?\b", en.lower())
                        and nes[:6].lower() not in es.lower()):
                    pend.append((h, j, en, nen, rol_es, rol_en))
        if not pend:
            return 0
        fuentes = []
        for _, _, en, nen, _, rol_en_ in pend:
            pat_frase = re.compile(_BB + re.escape(rol_en_) + "s?" + _BB, re.I)
            if pat_frase.search(en):
                fuentes.append(pat_frase.sub("ZQXROL", en))
            else:
                fuentes.append(re.sub(_BB + re.escape(nen) + "s?" + _BB, "ZQXROL",
                                      en, flags=re.I))
        x = self.tok_es(fuentes, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(DISPOSITIVO)
        with torch.no_grad():
            y = self.trad.generate(**x, num_beams=1, max_new_tokens=96)
        salida = self.tok_es.batch_decode(y, skip_special_tokens=True)
        arregladas = 0
        for (h, j, _, _, rol_es, _), t_ in zip(pend, salida):
            t_ = re.sub(r"ZQXROL(?:es|s)?", rol_es, t_, flags=re.I)
            if "ZQXROL" not in t_ and rol_es.split()[-1][:6].lower() in t_.lower():
                h["tareas_es"][j] = normalizar_verbo_es(glosario(t_))
                arregladas += 1
        return arregladas
    @torch.no_grad()
    @torch.no_grad()
    def _anclas(self, descripcion, roles, ej, k):
        """Una historia por funcion que la descripcion nombra, con la capacidad fijada
        como prefijo del decodificador y el beneficio escrito por el modelo. Se queda
        el beneficio que mejor puntua el reordenador entre N_BEN_ANCLA muestras."""
        if not roles:
            return []
        Er = self.sb.encode(roles, normalize_embeddings=True, show_progress_bar=False)
        vistas, out = set(), []
        vc = _func()["verbo_clase"]
        pares = []
        for suj, rg in rasgos_con_sujeto(descripcion):
            # «find and book study rooms»: dos historias pequenas mejor que una que pide
            # dos cosas (la I de INVEST). Solo si los dos lados son verbos.
            m = re.match("^([a-z]+) and ([a-z]+) (.+)$", rg)
            if m and m.group(1) in _VERBO_COORD and (m.group(2) in _VERBO_COORD
                                                     or m.group(2) in _VERBO_SUJ):
                pares += [(suj, f"{m.group(1)} {m.group(3)}"), (suj, f"{m.group(2)} {m.group(3)}")]
            else:
                pares.append((suj, rg))
        for i, (suj, rg) in enumerate(pares):
            if len(out) >= max(1, k // 2):
                break
            # un rasgo de una o dos palabras que nombra a un actor no es una funcion
            # («with access by roles for doctors and reception» -> «reception»)
            if len(rg.split()) <= 2 and any(rol_respaldado(rg, r) for r in roles):
                continue
            cap = capacidad_de_rasgo(rg, suj)
            if not cap or cap.lower() in vistas:
                continue
            vistas.add(cap.lower())
            # quien: el sujeto de la clausula si se parece a un actor del plan; si no,
            # el actor que el rasgo nombra; si no, el reparto en orden
            r = None
            if suj:
                for w in re.findall("[a-z]+", suj.lower()):
                    w = w[:-3] + "y" if w.endswith("ies") else w.rstrip("s")
                    r = next((x for x in _SUJETO_ROL.get(w, ()) if x in roles), None)
                    if r:
                        break
                r = r or next((x for x in roles if rol_respaldado(suj, x)), None)
            if suj and r is None:
                es = self.sb.encode([suj], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
                j = int(np.argmax(Er @ es))
                if float(Er[j] @ es) >= 0.45 or rol_respaldado(suj, roles[j]):
                    r = roles[j]
            if r is None and re.search('access|permission|roles' + _BB, cap.lower()):
                # el control de acceso por roles lo configura quien administra, no el
                # actor que la frase nombra («access by roles for doctors»)
                r = next((x for x in roles if re.search('admin|manager', x)), None)
            if r is None and _hay_transaccion(cap):
                # quien reserva, pide o paga es el que consume, no el que presta el
                # servicio: «book home cleaning services» casaba con «cleaner» por la raiz
                r = next((x for x in roles if x in _CONSUMO), None)
            if r is None:
                # el actor mas parecido a la capacidad; el reparto en orden daba
                # «Como paciente, quiero administrar las recetas»
                # se compara el OBJETO, sin el verbo: con «manage prescriptions» ganaba
                # siempre «clinic manager» por la palabra, no por la funcion
                obj = " ".join(cap.split()[1:]) or cap
                ec = self.sb.encode([obj], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
                r = roles[int(np.argmax(Er @ ec))]
            art = "an" if r[:1].lower() in "aeiou" else "a"
            pref = f"As {art} {r}, I want to {cap}, so that"
            ficha = ""
            if self.usa_funcion:
                ficha = " | funcion: " + vc.get(cap.split()[0].lower(), "manage")
            x = self.tok_a([f"feature -> user story | role: {r}{ficha} | {descripcion}{ej}"],
                           return_tensors="pt", truncation=True, max_length=256).to(DISPOSITIVO)
            ids = self.tok_a(pref, add_special_tokens=False, return_tensors="pt").input_ids
            dec = torch.cat([torch.full((1, 1), self.m_a.config.decoder_start_token_id),
                             ids], 1).to(DISPOSITIVO)
            g = self.m_a.generate(**x, decoder_input_ids=dec, max_new_tokens=40,
                                  do_sample=True, top_p=0.95, temperature=1.0,
                                  num_return_sequences=N_BEN_ANCLA)
            c = [s.strip() for s in self.tok_a.batch_decode(g, skip_special_tokens=True)]
            def _eco(s):
                tr = partir_historia(s)
                capw = {w[:5] for w in re.findall("[a-z]{3,}", tr[1].lower())}
                benw = {w[:5] for w in re.findall("[a-z]{3,}", tr[2].lower())
                        if w not in _RELLENO_BEN}
                return not (benw - capw)
            c = [s for s in c if PAT_HIST.match(s) and partir_historia(s)
                 and not beneficio_vacio(partir_historia(s)[2]) and not _eco(s)] or c
            if self.reordenador is not None and len(c) > 1:
                trip = [partir_historia(s) for s in c]
                ok = [j for j, t_ in enumerate(trip) if t_]
                if ok:
                    Ea = self.sb.encode([trip[j][1] for j in ok], normalize_embeddings=True,
                                        show_progress_bar=False)
                    Eb = self.sb.encode([trip[j][2] for j in ok], normalize_embeddings=True,
                                        show_progress_bar=False)
                    pr = self.reordenador.predict_proba(
                        np.hstack([Ea, Eb, Ea * Eb, np.abs(Ea - Eb)]).astype(np.float32))[:, 1]
                    c = [c[ok[int(np.argmax(pr))]]]
            out.append(c[0])
        return out

    def _vecinos(self, textos, E, tabla, col_hijos, k=K_VECINOS, n_hijos=4):
        q = self.sb.encode(textos, convert_to_tensor=True, normalize_embeddings=True,
                           show_progress_bar=False)
        v, i = torch.topk(q @ E.T, k, dim=1)
        ctx, proc = [], []
        for fila, sims in zip(i.cpu().numpy(), v.cpu().numpy()):
            partes, fuentes = [], []
            for j, s in zip(fila, sims):
                r = tabla.iloc[int(j)]
                hijos = list(r[col_hijos])[:n_hijos]
                titulo = (r.entrada if "entrada" in tabla.columns else r.entrada_en)[:80]
                partes.append(f"{titulo} => {' ; '.join(hijos)}")
                fuentes.append(dict(proyecto=str(r.proj), similitud=round(float(s), 3)))
            ctx.append(" || ".join(partes))
            proc.append(fuentes)
        return ctx, proc

    @torch.no_grad()
    def _generar(self, tok, m, textos, trozo=6, **kw):
        # chunked: 12 stories x beam 4 at once peaks too high when something else
        # (e.g. an Ollama model) is also holding VRAM
        out = []
        for a in range(0, len(textos), trozo):
            x = tok(textos[a:a + trozo], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(DISPOSITIVO)
            with torch.autocast(DISPOSITIVO, dtype=torch.bfloat16, enabled=DISPOSITIVO == "cuda"):
                g = m.generate(**x, **kw)
            out += tok.batch_decode(g, skip_special_tokens=True)
        return out

    def _colapsar(self, cands, umbral, k, consulta=None):
        cands = [c.strip() for c in cands if len(c.strip()) >= 15]
        en_fmt = [c for c in cands if PAT_HIST.match(c)]
        # el rol tiene que ser una persona, no un sustantivo cualquiera de la
        # descripcion; se muestrean 30 candidatas para quedarse con k, asi que
        # hay margen de sobra para descartar las que no lo cumplen
        con_rol = [c for c in en_fmt if rol_valido(c)]
        if len(con_rol) >= max(3, k // 2):
            cands = con_rol
        elif REPARAR_ROL and self.planificador is not None and consulta is not None:
            # Descartar no siempre sirve: hay descripciones donde CASI TODAS las
            # candidatas traen un rol invalido (el taller de coches, donde 7 de 8
            # decian "As a service appointment"), y entonces el filtro se rendia.
            # Aqui, en vez de rendirse, se SUSTITUYE el hueco del rol por un actor
            # del plantel del sector. No hace falta reentrenar nada: la capacidad
            # de la historia se conserva y solo cambia quien la pide.
            plantel = self.planificador(consulta, k=4)
            if plantel:
                arregladas, i = [], 0
                for c in en_fmt:
                    if rol_valido(c):
                        arregladas.append(c)
                    else:
                        arregladas.append(cambiar_rol(c, plantel[i % len(plantel)]))
                        i += 1
                cands = arregladas
            else:
                cands = en_fmt if len(en_fmt) >= 3 else cands
        else:
            cands = en_fmt if len(en_fmt) >= 3 else cands
        self.ultimas_cands = list(cands)   # para inspeccion: ver src/69
        self.ultimo_n_cands = len(cands)
        if not cands:
            return []
        E = self.sb.encode(cands, convert_to_tensor=True, normalize_embeddings=True,
                           show_progress_bar=False)
        S = (E @ E.T).float()
        # Seleccion por COBERTURA DE RASGOS, por omision. Medido sobre 30
        # descripciones reservadas (results/68_cobertura*.csv): los rasgos que la
        # descripcion enumera y quedan cubiertos por alguna tarea pasan del 81,1 %
        # al 87,8 %, y los backlogs que cubren todos, del 53,3 % al 66,7 %.
        # Cuesta 2,6 puntos de atribucion tarea->historia. Se acepta porque un
        # rasgo sin cubrir es trabajo que no existe en el backlog, mientras que
        # una tarea colgada de la historia equivocada sigue llevando su trabajo.
        # AGIX_COBERTURA=0 vuelve al orden por relevancia.
        p_ben = None
        if self.reordenador is not None and len(cands) > k:
            # Dos etapas: primero se reduce el banco a las 2k mejores por
            # coherencia del beneficio, y sobre esa mitad buena se optimiza la
            # cobertura de rasgos. Asi no se sacrifica una cosa por la otra: un
            # filtro duro dejaria pocas candidatas y la cobertura caeria.
            trip = [partir_historia(c) for c in cands]
            utiles = [i for i, t_ in enumerate(trip) if t_]
            if len(utiles) > k:
                caps = [trip[i][1] for i in utiles]
                bens = [trip[i][2] for i in utiles]
                Ea = self.sb.encode(caps, normalize_embeddings=True, show_progress_bar=False)
                Eb = self.sb.encode(bens, normalize_embeddings=True, show_progress_bar=False)
                Xr = np.hstack([Ea, Eb, Ea * Eb, np.abs(Ea - Eb)]).astype(np.float32)
                pr_u = self.reordenador.predict_proba(Xr)[:, 1]
                cands = [cands[i] for i in utiles]
                p_ben = pr_u
                E = self.sb.encode(cands, convert_to_tensor=True,
                                   normalize_embeddings=True, show_progress_bar=False)
                S = (E @ E.T).float()
        if os.environ.get("AGIX_COBERTURA", "1") == "1" and consulta is not None:
            trip_c = [partir_historia(c) for c in cands]
            idx = seleccion_por_rasgos(
                self.sb, cands, consulta, k, extra=p_ben,
                lam=float(os.environ.get("AGIX_LAM_BEN", 0.15)),
                mu=float(os.environ.get("AGIX_MU_ROL", 0.15)),
                roles=[rol_de(c) or "?" for c in cands],
                clases=([clase_funcion(c) or "?" for c in cands]
                        if self.usa_funcion else None),
                mu_clase=MU_CLASE,
                bens=[(re.sub(r"[^a-z ]", "", t[2].lower()).strip() if t else "")
                      for t in trip_c],
                nu=float(os.environ.get("AGIX_NU_BEN", 0.10)),
                caps=[(t[1] if t else "") for t in trip_c],
                nu_cap=NU_CAP, piso_cap=PISO_CAP, cupo_cap=CUPO_CAP,
                pena=self._pena_forma(trip_c))
            Sn = S.cpu().numpy()
            qe = self.sb.encode([consulta], convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=False)[0]
            rel = (E @ qe).float().cpu().numpy()
            return [dict(historia=cands[i], votos=int((Sn[i] >= umbral).sum()),
                         relevancia=round(float(rel[i]), 3)) for i in idx]
        if SUBMODULAR:
            # cobertura del banco de candidatas en lugar de colapsar y ordenar por
            # relevancia. Los votos se siguen calculando solo para la interfaz: el
            # consenso ya esta dentro del objetivo, no hace falta ordenar por el.
            Sn = S.cpu().numpy()
            rel = None
            if consulta is not None:
                qe = self.sb.encode([consulta], convert_to_tensor=True,
                                    normalize_embeddings=True, show_progress_bar=False)[0]
                rel = (E @ qe).float().cpu().numpy()
            idx = seleccion_submodular(Sn, k, rel=rel, lam=0.15)   # sin parada: se quieren k
            return [dict(historia=cands[i],
                         votos=int((Sn[i] >= umbral).sum()),
                         relevancia=round(float(rel[i]), 3) if rel is not None else 0.0)
                    for i in idx]
        usado = np.zeros(len(cands), bool)
        grupos = []
        for i in np.argsort([-len(c) for c in cands]):
            if usado[i]:
                continue
            msk = (S[i] >= umbral).cpu().numpy() & ~usado
            usado |= msk
            grupos.append((int(msk.sum()), cands[i]))
        if consulta is not None:      # rank by relevance to the description, not by votes alone
            qe = self.sb.encode([consulta], convert_to_tensor=True, normalize_embeddings=True,
                                show_progress_bar=False)[0]
            ge = self.sb.encode([t for _, t in grupos], convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=False)
            rel = (ge @ qe).cpu().numpy()
            orden = np.argsort(-rel)
            return [dict(historia=grupos[i][1], votos=grupos[i][0], relevancia=round(float(rel[i]), 3))
                    for i in orden[:k]]
        grupos.sort(key=lambda x: -x[0])
        return [dict(historia=t, votos=v) for v, t in grupos[:k]]

    # ------------------------------------------------------------ pipeline
    def __call__(self, descripcion, k_historias=K_HISTORIAS, n_muestras=N_MUESTRAS):
        T = {}
        descripcion_original = descripcion      # antes de traducirla; la usa el Sprint 0
        if SEMILLA is not None:
            # por llamada, no por proceso: asi el resultado de una descripcion no
            # depende de cuantas se generaron antes
            torch.manual_seed(int(SEMILLA))
            if DISPOSITIVO == "cuda":
                torch.cuda.manual_seed_all(int(SEMILLA))
        act_es = []
        if self.trad_es_en is not None and es_espanol(descripcion):
            # paso 4c: los actores se leen de la FUENTE, antes de que el traductor los
            # pueda romper («apoderados» -> «agents», «equipo» -> «computer»)
            act_es = actores_es_nombrados(descripcion)
            self.ultimos_actores_es = act_es
            t = time.time()
            x = self.tok_ee([descripcion], return_tensors="pt", truncation=True,
                            max_length=256).to(DISPOSITIVO)
            with torch.no_grad():
                y = self.trad_es_en.generate(**x, num_beams=1, max_new_tokens=220)
            descripcion = self.tok_ee.batch_decode(y, skip_special_tokens=True)[0]
            T["traducir_entrada"] = time.time() - t
        t = time.time()
        ctx, proc_a = self._vecinos([descripcion], self.EA, self.A, "hijos")
        T["recuperar_arriba"] = time.time() - t

        t = time.time()
        ej = f" | examples: {ctx[0]}" if EJEMPLOS_ARRIBA else " | examples: "
        gen_a = dict(max_new_tokens=64, do_sample=True, top_p=0.95, temperature=1.0)
        roles = (self.planificador(descripcion, previos=act_es)
                 if self.planificador is not None else [])
        # para inspeccion y para la puerta de entrada: lo que vio el planificador
        self.ultima_descripcion_en = descripcion
        self.ultimo_plan = list(roles)
        # Con la ficha de rol la diversidad ya la da la lista de actores, asi que
        # se puede dejar de muestrear. Leyendo las salidas del metodo A se vio
        # que el muestreo a temperatura 1,0 producia 12 historias sin ningun
        # acuerdo entre muestras (votos max 1): mucha entropia y poca precision.
        # AGIX_ARRIBA_HAZ=1 genera con haz por rol, que es determinista.
        if roles and os.environ.get("AGIX_ARRIBA_HAZ") == "1":
            gen_a = dict(max_new_tokens=64, do_sample=False, num_beams=None)
        pares_fun = []
        if roles and self.usa_funcion:
            pares_fun = plan_funciones(roles, k_historias, self.sb)
        if pares_fun:
            # una tanda por PAR (actor, clase de funcion): asi el banco trae las seis
            # clases y la seleccion puede repartirlas, en lugar de recibir solo view
            por_par = max(6, n_muestras // max(1, len(pares_fun)))
            cands, por_rol_cands = [], []
            for r, cl in pares_fun:
                c = self._generar(
                    self.tok_a, self.m_a,
                    [f"feature -> user story | role: {r} | funcion: {cl} | {descripcion}{ej}"],
                    num_return_sequences=por_par, **gen_a)
                por_rol_cands.append(c)
                cands += c
        elif roles:
            # una tanda de muestras por actor, y su cuota de historias: el
            # backlog sale con varios roles sin depender de la suerte
            # Con la ficha se muestrea por actor, asi que cada tanda es mas
            # pequena y el deduplicador deja menos: pidiendo 12 historias salian
            # 6. Se muestrea mas por rol para que sobrevivan las 12; el coste es
            # lineal en tiempo de generacion de la mitad de arriba, que es el
            # 16 % del total.
            por_rol = max(12, n_muestras // len(roles))
            if gen_a.get("num_beams", 1) is None:      # haz: una viga por historia pedida
                por_rol = max(2, -(-k_historias // len(roles)))
                gen_a = {**gen_a, "num_beams": por_rol, "no_repeat_ngram_size": 4}
            cands, por_rol_cands = [], []
            for r in roles:
                c = self._generar(self.tok_a, self.m_a,
                                  [f"feature -> user story | role: {r} | {descripcion}{ej}"],
                                  num_return_sequences=por_rol, **gen_a)
                por_rol_cands.append(c)
                cands += c
        else:
            cands = self._generar(self.tok_a, self.m_a,
                                  [f"feature -> user story: {descripcion}{ej}"],
                                  num_return_sequences=n_muestras, **gen_a)
        T["generar_historias"] = time.time() - t

        t = time.time()
        if (roles or pares_fun) and not BANCO_GLOBAL:
            cuota = max(1, k_historias // len(roles))
            historias = []
            for c in por_rol_cands:
                historias += self._colapsar(c, UMBRAL_DUP, cuota, descripcion)
            historias = self._colapsar([h["historia"] for h in historias],
                                       UMBRAL_DUP, k_historias, descripcion)
        elif roles or pares_fun:
            # Sin cuota por rol: se juntan TODAS las candidatas condicionadas por
            # actor y se selecciona una sola vez sobre el banco entero. La cuota
            # forzaba 3 historias por actor tenga o no 3 cosas que decir, y tiraba
            # candidatas antes de que la seleccion pudiera optimizar la cobertura
            # del alcance. La variedad de roles se conserva igual, porque las
            # candidatas ya se generaron condicionadas a cada actor.
            self.ultimo_banco = [c for lista in por_rol_cands for c in lista]
            historias = self._colapsar(self.ultimo_banco, UMBRAL_DUP,
                                       k_historias, descripcion)
        else:
            historias = self._colapsar(cands, UMBRAL_DUP, k_historias, descripcion)
        T["deduplicar_historias"] = time.time() - t
        self.ultimas_anclas = []
        if ANCLAR and roles:
            t = time.time()
            anclas = self._anclas(descripcion, roles, ej, k_historias)
            self.ultimas_anclas = anclas
            if anclas:
                Ea = self.sb.encode(anclas, normalize_embeddings=True, show_progress_bar=False)
                resto = list(historias)
                if resto:
                    Eh = self.sb.encode([h["historia"] for h in resto],
                                        normalize_embeddings=True, show_progress_bar=False)
                    lejos = (Eh @ Ea.T).max(1) < UMBRAL_DUP
                    resto = [h for h, l in zip(resto, lejos) if l]
                plan = {r.lower() for r in roles}
                dentro = [h for h in resto if (rol_de(h["historia"]) or "").lower() in plan]
                fuera = [h for h in resto if (rol_de(h["historia"]) or "").lower() not in plan]
                historias = ([dict(historia=a_, votos=1, relevancia=1.0, origen="ancla")
                              for a_ in anclas] + dentro + fuera)[:k_historias]
            T["anclar"] = time.time() - t
        if not historias:
            return dict(historias=[], tiempos=T)

        t = time.time()
        textos = [h["historia"] for h in historias]
        ctxb, proc_b = self._vecinos(textos, self.EB, self.B, "tareas")
        T["recuperar_abajo"] = time.time() - t

        t = time.time()
        # El haz maximiza verosimilitud, y con eso amplifica la formulacion mas
        # probable: medido, el 83,6 % de las tareas generadas empiezan por uno de
        # solo cinco arranques de dos palabras, cuando las tareas escritas por
        # personas en el corpus de Jira estan en el 2,6 % y el profesor LLM en el
        # 28 %. AGIX_ABAJO_MUESTREO=1 cambia el haz por muestreo para recuperar
        # variedad de formulacion sin tocar los datos.
        if GEN_ABAJO is not None:          # lo pone src/64_decodificador.py para barrer
            gen = dict(max_new_tokens=192, no_repeat_ngram_size=4,
                       repetition_penalty=1.3, **GEN_ABAJO)
        elif os.environ.get("AGIX_ABAJO_MUESTREO") == "1":
            gen = dict(max_new_tokens=192, do_sample=True, top_p=0.92, temperature=0.9,
                       no_repeat_ngram_size=4, repetition_penalty=1.3)
        else:
            gen = dict(max_new_tokens=192, num_beams=BEAMS_ABAJO, no_repeat_ngram_size=4,
                       repetition_penalty=1.3)
        srcs = [f"decompose story into tasks: {h} | examples: {c if EJEMPLOS_ABAJO else ''}"
                for h, c in zip(textos, ctxb)]
        listas = [self._generar(self.tok_b, self.m_b, srcs, **gen)]
        if self.m_b2 is not None:
            listas.append(self._generar(self.tok_b, self.m_b2, srcs, **gen))
        if self.m_b3 is not None:
            listas.append(self._generar(self.tok_b, self.m_b3, srcs, **gen))
        if self.conjunto == "completo":      # one prompt per retrieved neighbour
            for j in (0, -1):
                sj = [f"decompose story into tasks: {h} | examples: {c.split(' || ')[j]}"
                      for h, c in zip(textos, ctxb)]
                listas.append(self._generar(self.tok_b, self.m_b, sj, **gen))
        if VETAR_ARRANQUES and listas:
            # Segunda pasada: se cuentan los arranques de la primera y se VETAN los
            # mas repetidos, para que el banco traiga alternativas. Sin esto la
            # seleccion no puede variar la formulacion porque todas las candidatas
            # empiezan igual.
            cuenta = collections.Counter(
                a for lst in listas for s_ in lst for x in separar_tareas(s_)
                if (a := arranque(x)))
            top = [a for a, n in cuenta.most_common(3) if n >= 3]
            if top:
                mal = []
                for a in top:
                    for var in (a, a.capitalize(), a[0].upper() + a[1:]):
                        ids = self.tok_b(var, add_special_tokens=False).input_ids
                        if ids:
                            mal.append(ids)
                if mal:
                    listas.append(self._generar(self.tok_b, self.m_b, srcs,
                                                bad_words_ids=mal, **gen))
        salidas = [" ; ".join(x) for x in zip(*listas)] if len(listas) > 1 else listas[0]
        T["generar_tareas"] = time.time() - t

        t = time.time()
        for h, s, f in zip(historias, salidas, proc_b):
            h["tareas"] = [t_ for t_ in separar_tareas(s)
                           if not tarea_degenerada(t_, h["historia"])]
            h["origen"] = f
        # duplicate tasks: across stories AND inside one story. The two generators
        # of the ensemble answer the same prompt, so inside a story they often
        # return near-copies ("Create a multi mode" / "Create an multi mode");
        # only exact text was being removed, leaving 30 % of the tasks redundant.
        # Measured in src/39: at 0.95 it drops 17 % of the tasks and costs 0.0023
        # of soft-F1, and it is free -- this similarity matrix was computed anyway.
        if SUBMODULAR:
            # Por historia: cobertura del banco de tareas que produjeron los
            # generadores del conjunto, con regla de parada. El numero de tareas
            # lo decide el contenido en lugar de ser una constante, que es lo que
            # hoy hace que una historia trivial y una grande reciban las mismas 3.
            for h in historias:
                ts = h["tareas"]
                if len(ts) < 2:
                    continue
                Et = self.sb.encode(ts, convert_to_tensor=True, normalize_embeddings=True,
                                    show_progress_bar=False)
                St = (Et @ Et.T).float().cpu().numpy()
                idx = seleccion_submodular(St, len(ts), tau=TAU_CORTE)
                h["tareas"] = [ts[i] for i in sorted(idx)]
            # y entre historias se sigue quitando la repeticion literal, que ahi
            # si puede ser trabajo real duplicado y no ruido del muestreo
            vistas, arr = {}, collections.Counter()
            for h in historias:
                keep = []
                for t_ in h["tareas"]:
                    k2 = re.sub(r"[^a-z0-9]+", " ", t_.lower()).strip()
                    a = arranque(t_)
                    if k2 in vistas:
                        continue
                    if TOPE_ARRANQUE_SALIDA and a and arr[a] >= TOPE_ARRANQUE_SALIDA:
                        continue          # ya hay suficientes tareas con ese arranque
                    vistas[k2] = True
                    if a:
                        arr[a] += 1
                    keep.append(t_)
                # Nunca dejar la historia sin tareas. Leyendo la salida de 10 dominios
                # se vio que el 3,3 % de las historias acaba sin ninguna tarea y el
                # 8,3 % con una sola: el tope de arranques y el deduplicador pueden
                # vaciar una historia entera, y el servidor entonces la descarta, asi
                # que el backlog encoge de 12 a 11 sin decir nada. Una historia con
                # una tarea repetida se lee peor que una historia sin nada, pero una
                # historia que desaparece es peor que las dos.
                # El rescate prefiere una tarea que NO este ya en otra historia: con la
                # primera a ciegas salian tareas literalmente repetidas entre historias
                # (prueba de extremo a extremo: las 2 de 36 backlogs venian de aqui).
                if not keep and h["tareas"]:
                    libre = [t_ for t_ in h["tareas"] if re.sub("[^a-z0-9]+", " ",
                             t_.lower()).strip() not in vistas]
                    keep = (libre or h["tareas"])[:1]
                    vistas[re.sub("[^a-z0-9]+", " ", keep[0].lower()).strip()] = True
                h["tareas"] = keep
            T["deduplicar_tareas"] = time.time() - t
            todas = []
        else:
            todas = [(i, j, t_) for i, h in enumerate(historias)
                     for j, t_ in enumerate(h["tareas"])]
        if todas:
            E = self.sb.encode([x[2] for x in todas], convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False)
            S = (E @ E.T).float().cpu().numpy()
            Sv = None
            if UMBRAL_TAREA_SINV:
                sinv = [" ".join(x[2].split()[1:]) or x[2] for x in todas]
                Ev = self.sb.encode(sinv, convert_to_tensor=True,
                                    normalize_embeddings=True, show_progress_bar=False)
                Sv = (Ev @ Ev.T).float().cpu().numpy()
            quitar = set()
            for a in range(len(todas)):
                if a in quitar:
                    continue
                for b in range(a + 1, len(todas)):
                    if b in quitar:
                        continue
                    # mas exigente dentro de una misma historia: ahi una casi-copia
                    # es siempre ruido, mientras que entre historias puede ser trabajo real
                    umbral = (UMBRAL_TAREA_DUP if todas[a][0] != todas[b][0]
                              else UMBRAL_TAREA_DUP_INTRA)
                    misma = todas[a][0] == todas[b][0]
                    if S[a, b] >= umbral or (
                            misma and Sv is not None and Sv[a, b] >= UMBRAL_TAREA_SINV):
                        quitar.add(b)
            fuera = {(todas[b][0], todas[b][1]) for b in quitar}
            for i, h in enumerate(historias):
                keep = [t_ for j, t_ in enumerate(h["tareas"]) if (i, j) not in fuera]
                h["tareas"] = keep or h["tareas"][:1]    # nunca vaciar la historia
        # Tope de arranques repetidos en TODO el backlog. Va aqui, despues de
        # deduplicar: el deduplicador semantico no ve la plantilla, porque
        # "Store the price" y "Store the schedule" son distintas y las dos
        # legitimas; lo que sobra es la tercera y la cuarta.
        if TOPE_ARRANQUE_SALIDA:
            arr = collections.Counter()
            for h in historias:
                keep = []
                for t_ in h["tareas"]:
                    a = arranque(t_)
                    if a and arr[a] >= TOPE_ARRANQUE_SALIDA:
                        continue
                    if a:
                        arr[a] += 1
                    keep.append(t_)
                h["tareas"] = keep or h["tareas"][:1]    # nunca vaciar la historia
        T["deduplicar_tareas"] = time.time() - t

        # Reasignar la tarea a la historia a la que de verdad pertenece. La mitad de
        # abajo genera las tareas de cada historia por separado, asi que nada impide
        # que la tarea de una acabe describiendo el trabajo de otra; medido sobre dos
        # backlogs recien generados, la atribucion (la tarea se parece mas a SU
        # historia que a las hermanas) era del 73 % en banca y del 56 % en bodas,
        # contra un 8 % de azar. Mover es mejor que descartar: la tarea lleva trabajo
        # real, solo esta colgada del sitio equivocado.
        #
        # Se hace AQUI, antes de traducir, porque medido en espanol la atribucion
        # esta corrompida por la traduccion: una historia que dice "ley recurrente"
        # (mala traduccion de "recurring bill") ya no se parece a sus propias tareas,
        # que dicen "factura", y la reasignacion las mandaba a otra historia.
        if REASIGNAR_TAREAS and len(historias) > 1:
            t = time.time()
            plano = [(i, t_) for i, h in enumerate(historias) for t_ in h["tareas"]]
            if plano:
                Eh = self.sb.encode([h["historia"] for h in historias],
                                    normalize_embeddings=True, show_progress_bar=False)
                Et = self.sb.encode([x[1] for x in plano], normalize_embeddings=True,
                                    show_progress_bar=False)
                S = Et @ Eh.T
                destino = [i for i, _ in plano]
                cand = []
                for j, (i, _) in enumerate(plano):
                    b = int(np.argmax(S[j]))
                    if b != i:
                        cand.append((float(S[j, b] - S[j, i]), j, i, b))
                # de mayor margen a menor, y nunca se vacia una historia: una
                # historia sin tareas se lee peor que una tarea mal colgada
                quedan = collections.Counter(i for i, _ in plano)
                movidas = 0
                for m, j, i, b in sorted(cand, reverse=True):
                    if m <= MARGEN_REASIGNAR or quedan[i] <= 1:
                        continue
                    destino[j] = b
                    quedan[i] -= 1
                    quedan[b] += 1
                    movidas += 1
                # Tarea AJENA: no se parece a ninguna historia del backlog. Es la
                # regla generica que el lexico de Jira no puede dar, porque las
                # fugas que quedan no comparten vocabulario ("Create a source
                # distribution for the release", "Add metrics for the number of
                # recovered queues"). Medido sobre 78 tareas de dos backlogs: el
                # percentil 5 de la similitud maxima es 0,409 y las tres por debajo
                # de 0,30 son justo esas fugas, con 0,06 de margen hasta la primera
                # tarea legitima ("Store the freeze data in the database", 0,363).
                fuera_piso = set()
                if PISO_TAREA:
                    for j, (i, _) in enumerate(plano):
                        if float(S[j].max()) < PISO_TAREA and quedan[destino[j]] > 1:
                            fuera_piso.add(j)
                            quedan[destino[j]] -= 1
                if movidas or fuera_piso:
                    nuevas = [[] for _ in historias]
                    for j, (_, t_) in enumerate(plano):
                        if j not in fuera_piso:
                            nuevas[destino[j]].append(t_)
                    for h, ts in zip(historias, nuevas):
                        h["tareas"] = ts
                self.ultimas_movidas = movidas
                self.ultimas_ajenas = len(fuera_piso)
            T["reasignar_tareas"] = time.time() - t

        if self.trad is not None:
            t = time.time()
            plano = textos + [t_ for h in historias for t_ in h["tareas"]]
            prot = [proteger(t_) for t_ in plano]
            x = self.tok_es([p for p, _ in prot], return_tensors="pt", padding=True,
                            truncation=True, max_length=128).to(DISPOSITIVO)
            with torch.no_grad():
                y = self.trad.generate(**x, num_beams=1, max_new_tokens=96)
            es = [glosario(restaurar(t_, f)) for t_, (_, f) in
                  zip(self.tok_es.batch_decode(y, skip_special_tokens=True), prot)]
            # y el glosario con la fuente inglesa, alineado indice a indice con `plano`
            es = [glosario_fuente(e, en_, descripcion) for e, en_ in zip(es, plano)]
            es = [glosario_general(e, en_) for e, en_ in zip(es, plano)]
            it = iter(es[len(textos):])
            for h, e in zip(historias, es[:len(textos)]):
                h["historia_es"] = e
                h["tareas_es"] = [normalizar_verbo_es(next(it)) for _ in h["tareas"]]
            if UNIF_ROL_ES:
                self.ultimos_cambios_rol = unificar_roles_es(historias)
            if REPARAR_ROL_TAREA:
                self.ultimas_tareas_rol = self._reparar_rol_en_tareas(historias)
            if REPARAR_VERBO_TAREA:
                self.ultimas_tareas_verbo = self._reparar_verbo_en_tareas(historias)
            if ECO_RAIZ:
                # El eco por flexion solo se ve en espanol: «Notificar ... una
                # notificacion» comparte raiz en espanol y no en ingles («notify» /
                # «notification»), y el filtro de candidatas corre ANTES de traducir.
                # Se quitan las dos listas a la vez para que sigan emparejadas, y nunca
                # se deja una historia sin tareas. Medido: 5 de 252 tareas.
                n_eco = 0
                for h in historias:
                    tes = h.get("tareas_es") or []
                    if len(tes) < 2:
                        continue
                    keep = [j for j, t_ in enumerate(tes) if not eco_por_raiz(t_)]
                    if not keep:
                        keep = [0]
                    if len(keep) < len(tes):
                        n_eco += len(tes) - len(keep)
                        h["tareas_es"] = [tes[j] for j in keep]
                        h["tareas"] = [h["tareas"][j] for j in keep
                                       if j < len(h["tareas"])] or h["tareas"][:1]
                self.ultimas_tareas_eco = n_eco
            T["traducir"] = time.time() - t

        sprint0 = []
        if SPRINT0:
            sprint0 = plan_sprint0(descripcion, descripcion_original, roles)
        T["total"] = sum(T.values())
        return dict(historias=historias, tiempos={k: round(v, 3) for k, v in T.items()},
                    # bloque aparte, ver plan_sprint0; vacio si AGIX_SPRINT0 no esta a 1
                    sprint0=sprint0,
                    n_tareas_sprint0=sum(len(h["tareas"]) for h in sprint0),
                    vecinos_proyecto=proc_a[0],
                    # cuantas candidatas se muestrearon de verdad, para que la
                    # interfaz no invente el denominador: con ficha de rol son
                    # max(12, n/roles) por actor, no las n que se piden
                    n_banco=int(getattr(self, "ultimo_n_cands", 0)),
                    n_tareas=sum(len(h["tareas"]) for h in historias))


DEMOS = ["Una aplicación web para reservar servicios de limpieza a domicilio, con citas recurrentes y pagos en la app.",
         "A mobile app where university students can find and book study rooms on campus in real time.",
         "Plataforma para que pequeños agricultores registren sus cosechas y vendan directamente a restaurantes."]

if __name__ == "__main__":
    traducir = "--traducir" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    p = Pipeline(traducir=traducir)
    print(f"carga del pipeline: {p.carga_s}s  (pool arriba {len(p.A):,}, pool abajo {len(p.B):,})")
    textos = DEMOS if ("--demo" in sys.argv or not args) else args
    salidas = []
    for d in textos:
        r = p(d)
        salidas.append(dict(descripcion=d, **r))
        print(f"\n=== {d}\n  tiempos {r['tiempos']}  historias {len(r['historias'])}  tareas {r['n_tareas']}")
        print(f"  vecinos: {r['vecinos_proyecto']}")
        for h in r["historias"][:5]:
            print(f"   [{h['votos']:>2} votos] {h['historia']}")
            for t_ in h["tareas"][:4]:
                print(f"        - {t_}")
    json.dump(salidas, open(f"{BASE}/results/19_demo.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
