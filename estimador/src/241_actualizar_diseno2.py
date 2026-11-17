"""Segunda tanda sobre DISENO_PIPELINE.md: enganche, contrato, metricas y
la seccion nueva de lo que se descarto.

Lo que se descarto importa tanto como lo que se puso: evita que quien implemente
en produccion gaste semanas en vias que ya estan medidas y cerradas.
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
DOC = Path("DISENO_PIPELINE.md")


def pc(v, d=1):
    return f"{v * 100:.{d}f} %".replace(".", ",")


def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pon(txt, **kw):
    for k, v in kw.items():
        txt = txt.replace(f"@@{k}@@", str(v))
    assert "@@" not in txt, "quedan tokens sin sustituir"
    return txt


def main():
    s = io.open(DOC, encoding="utf-8").read()
    anc = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    rej = pd.read_csv("results/rejilla_valores/contra_literatura.csv").set_index("modelo")
    lit = pd.read_csv("results/baselines_literatura.csv").set_index("modelo")
    art = pd.read_csv("results/centrar_artefacto/resumen.csv").set_index("salida")
    cor = art.loc["E · B + rejilla, SIN centrar"]
    hoy = art.loc["A · como se despliega hoy"]
    rep = []

    # ------------------------------------------------- B.6.4 -> B.6.6, estilo
    s = s.replace("### B.6.4 · Enganche con el generador de backlog",
                  "### B.6.6 · Enganche con el generador de backlog", 1)
    viejo = """**2 · El factor de estilo del total NO es el de la mediana (E74).** La cabeza A
multiplica por 0,8313 para compensar que el texto del generador es mas escueto que un
titulo de Jira."""
    assert viejo in s
    s = s.replace(viejo, """**2 · El factor de estilo es CONDICIONAL al origen del texto, y cada cabeza tiene el
suyo.** Se midió sobre el texto del generador, que es más escueto que un título de Jira.
Sobre texto escrito por una persona el sesgo medido era ×1,002, o sea ninguno: aplicarlo ahí
solo encoge la predicción.

```python
est.estimar_tareas(textos, estilo="persona")     # por omisión: NO aplica factor
est.estimar_tareas(textos, estilo="generador")   # aplica 0,8313
```

Durante un tiempo se aplicó siempre, y el artefacto mostraba un **31 % menos de horas** de las
reales (sesgo ×0,820 en lugar de 1,0). Ninguna métrica del proyecto lo veía, porque todas se
calculan sobre un modelo limpio y no sobre el paquete. Corregido:

| 2.000 tareas reales de prueba | ±25 % | factor 2 | MAE | SA | sesgo |
|---|---|---|---|---|---|
| aplicando siempre el factor | @@p_hoy@@ | @@f_hoy@@ | @@m_hoy@@ | @@s_hoy@@ | ×@@b_hoy@@ |
| **condicional (lo correcto)** | **@@p_cor@@** | **@@f_cor@@** | **@@m_cor@@** | **@@s_cor@@** | **×@@b_cor@@** |

> Dejar el sesgo dentro daba **más** ±25 % (18,9 % contra 17,5 %), porque la métrica premia
> subestimar. Se eligió la salida insesgada: para una herramienta de planificación, mostrar un
> 17 % menos de horas haría que los equipos planificaran corto de forma sistemática.

**El de la cabeza B es otro (E74).**""", 1)
    s = pon(s, p_hoy=pc(hoy.PRED25), f_hoy=pc(hoy.factor2), m_hoy=n(hoy.MAE_h, 2),
            s_hoy=n(hoy.SA, 4), b_hoy=n(hoy.sesgo, 3),
            p_cor=pc(cor.PRED25), f_cor=pc(cor.factor2), m_cor=n(cor.MAE_h, 2),
            s_cor=n(cor.SA, 4), b_cor=n(cor.sesgo, 3))
    rep.append("B.6.6 factor condicional")

    # ---------------------------------------------------------------- B.7
    ini = s.index("## B.7 · Contrato de la interfaz")
    fin = s.index("## B.8 · Artefactos y rendimiento medido")
    s = s[:ini] + """## B.7 · Contrato de la interfaz

```python
est = EstimadorV6()                      # carga models/estimador_pbe_v6.joblib

est.estimar_tareas(textos,               # -> DataFrame
                   estilo="persona",     #   o "generador" si viene de la tubería 1
                   personas=None,        #   quién hará cada tarea; None donde no se sepa
                   historial=None,       #   {persona: [(texto, horas_reales), ...]}
                   estimaciones=None)    #   lo que estimó el usuario; None donde no
#   texto · horas · valor_probable · banda_baja · banda_alta
#   ajuste_persona · con_estimacion

est.estimar_proyecto(textos,             # -> dict
                     cerradas=None,      #   horas REALES ya registradas; None donde no
                     estimaciones=None,  #   lo que estimó el usuario; None donde no
                     nivel=0.8)
#   total_horas · horas_cerradas · horas_abiertas
#   banda_baja · banda_alta · nivel · tareas · cerradas
#   con_estimacion · lambda_usada · tramo_banda · k_banda · contraccion_tramo

est.tira_probabilidad(textos,            # -> lista por tarea de (desde_h, hasta_h, prob)
                      nivel=None, max_trozos=None)
```

Reglas del contrato, que importan:

- **El texto se espera en inglés**, solo el **título**, sin descripción. El generador de PBE ya
  produce inglés internamente y traduce solo para mostrar. Pasarle el español no falla: solo
  acierta menos.
- `cerradas`, `estimaciones` y `personas` son listas **de la misma longitud que `textos`**, con
  `None` en las posiciones que no aplican.
- `horas` es la mediana; `valor_probable` es un valor de la rejilla. **Ninguna de las dos es
  para sumar**: el total sale de `estimar_proyecto`, que usa la otra cabeza.
- `historial` solo debe traer tareas **del mismo proyecto** (ver B.6.4).
- Los campos de diagnóstico (`lambda_usada`, `tramo_banda`, `k_banda`,
  `contraccion_tramo`, `ajuste_persona`, `con_estimacion`) existen para que la interfaz pueda
  explicar **de dónde sale** cada número, que es un criterio de aceptación de las historias de
  usuario.

### Lo que la interfaz debería mostrar

| situación | qué enseñar |
|---|---|
| backlog recién creado | total con su banda, marcado como provisional |
| el usuario estima tareas | indicar qué % lleva estimado y que el total mejora al completar |
| hay tareas cerradas | cuántas anclan el total, y que la banda se estrecha por eso |
| detalle de una tarea | horas, valor probable, banda, y opcionalmente la tira |
| tarea asignada | el factor de la persona y cuántas cerradas suyas lo sostienen |

---

""" + s[fin:]
    rep.append("B.7 contrato")

    # ---------------------------------------------------------------- B.9
    ini = s.index("## B.9 · Calidad medida")
    fin = s.index("## B.10 · Límite conocido")
    s = s[:ini] + pon("""## B.9 · Calidad medida

### Total del proyecto — *la métrica que se presenta*

114 proyectos, **todos predichos como si fueran nuevos**:

| | MdMRE | ±25 % | factor 2 | MAE (h) |
|---|---|---|---|---|
| **en frío** | **@@m0@@** | **@@p0@@** | @@f0@@ | @@a0@@ |
| con 3 tareas cerradas | @@m3@@ | @@p3@@ | @@f3@@ | @@a3@@ |
| con 5 tareas cerradas | @@m5@@ | @@p5@@ | @@f5@@ | @@a5@@ |
| con 10 tareas cerradas | @@m10@@ | @@p10@@ | @@f10@@ | @@a10@@ |
| con 20 tareas cerradas | @@m20@@ | @@p20@@ | @@f20@@ | @@a20@@ |

**Con estimación del usuario** (43 proyectos que la traen en ≥80 % de sus tareas):

| | ±25 % |
|---|---|
| en frío | 46,3 % |
| el usuario estima **todo** el backlog | 73,2 % |
| **+ 10 cerradas con λ=96** | **87,4 %** |

> **Cuidado con el 87,4 %.** Necesita **tres condiciones a la vez**: que el proyecto tenga
> estimación del usuario, que esté estimado *todo* el backlog, y que además haya 10 tareas
> cerradas. No es lo que da el sistema a un proyecto nuevo. La cifra de titular es **@@p0@@ en
> frío**, subiendo a **@@p20@@ con 20 cerradas**.

### Por tarea — bloque de prueba, 4.225 tareas

| | MdMRE | ±25 % | MAE (h) |
|---|---|---|---|
| línea base (mediana por proyecto) | @@lb_m@@ | @@lb_p@@ | @@lb_a@@ |
| Deep-SE (LSTM) | @@ds_m@@ | @@ds_p@@ | @@ds_a@@ |
| GPT2SP (GPT-2 ajustado) | @@gp_m@@ | @@gp_p@@ | @@gp_a@@ |
| **nuestro, ajustado a la rejilla** | @@nu_m@@ | **@@nu_p@@** | @@nu_a@@ |
| *una constante de 2,45 h* | *@@ct_m@@* | *@@ct_p@@* | *@@ct_a@@* |

Dos lecturas, y las dos hay que darlas:

**Estamos por encima de lo publicado**, y con un protocolo **más estricto**: nuestras
predicciones son fuera de pliegue agrupando por **proyecto entero**, así que el modelo no vio ni
la tarea ni su proyecto; las reimplementaciones entrenan con tareas de los mismos proyectos.

**Pero una constante le gana a todo.** No es un resultado sobre el modelo sino sobre la métrica:
PRED(25) mide la dispersión y la curtosis de estimado/real, no la capacidad de ordenar
(Shepperd y Kitchenham). Por eso se reporta **siempre junto al SA y al Spearman**, donde esa
constante saca −0,015 y 0,000 y el modelo no.

### Lo que gana la suma

Las **mismas** predicciones de la cabeza B, medidas en las dos unidades sobre los mismos 114
proyectos:

| unidad | n | MdMRE | ±25 % |
|---|---|---|---|
| por tarea | 24.622 | 1,370 | 12,1 % |
| **por proyecto** | 114 | **@@m0@@** | **@@p0@@** |

**Sumar divide el MdMRE por 3,86.** Que la cabeza de la media salga mal por tarea no es un
defecto: con un objetivo asimétrico la media está por encima de la mediana, así que sobreestima
la tarea típica a propósito. Es el precio de poder sumarla, y es lo que justifica **dos cabezas
y no una**.

---

""", m0=n(anc.loc[0, "MdMRE"]), p0=pc(anc.loc[0, "PRED25"]),
        f0=pc(anc.loc[0, "factor2"]), a0=n(anc.loc[0, "MAE_h"], 0),
        m3=n(anc.loc[3, "MdMRE"]), p3=pc(anc.loc[3, "PRED25"]),
        f3=pc(anc.loc[3, "factor2"]), a3=n(anc.loc[3, "MAE_h"], 0),
        m5=n(anc.loc[5, "MdMRE"]), p5=pc(anc.loc[5, "PRED25"]),
        f5=pc(anc.loc[5, "factor2"]), a5=n(anc.loc[5, "MAE_h"], 0),
        m10=n(anc.loc[10, "MdMRE"]), p10=pc(anc.loc[10, "PRED25"]),
        f10=pc(anc.loc[10, "factor2"]), a10=n(anc.loc[10, "MAE_h"], 0),
        m20=n(anc.loc[20, "MdMRE"]), p20=pc(anc.loc[20, "PRED25"]),
        f20=pc(anc.loc[20, "factor2"]), a20=n(anc.loc[20, "MAE_h"], 0),
        lb_m=n(lit.loc["linea base (mediana por proyecto)"].MdMRE),
        lb_p=pc(lit.loc["linea base (mediana por proyecto)"].PRED25),
        lb_a=n(lit.loc["linea base (mediana por proyecto)"].MAE_h, 2),
        ds_m=n(lit.loc["Deep-SE (LSTM)"].MdMRE),
        ds_p=pc(lit.loc["Deep-SE (LSTM)"].PRED25),
        ds_a=n(lit.loc["Deep-SE (LSTM)"].MAE_h, 2),
        gp_m=n(lit.loc["GPT2SP (GPT-2 ajustado)"].MdMRE),
        gp_p=pc(lit.loc["GPT2SP (GPT-2 ajustado)"].PRED25),
        gp_a=n(lit.loc["GPT2SP (GPT-2 ajustado)"].MAE_h, 2),
        nu_m=n(rej.loc["nuestro, AJUSTADO a la rejilla"].MdMRE),
        nu_p=pc(rej.loc["nuestro, AJUSTADO a la rejilla"].PRED25),
        nu_a=n(rej.loc["nuestro, AJUSTADO a la rejilla"].MAE_h, 2),
        ct_m=n(rej.loc["la mejor constante"].MdMRE),
        ct_p=pc(rej.loc["la mejor constante"].PRED25),
        ct_a=n(rej.loc["la mejor constante"].MAE_h, 2)) + s[fin:]
    rep.append("B.9 metricas")

    io.open(DOC, "w", encoding="utf-8").write(s)
    print("reescrito: " + " · ".join(rep))
    print(f"{DOC}: {len(s):,} caracteres")


if __name__ == "__main__":
    main()
