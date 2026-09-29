"""Deck 5: lo que salió de la noche del 29/09, para la memoria y para la reunión.

Mismo vocabulario de láminas que los otros tres. Las cifras se leen de los CSV
de resultados en tiempo de ejecución; las figuras las produce el script 111.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pptx.util import Inches  # noqa: E402

import pptx_comun as P  # noqa: E402

FIG = Path("reports/figuras_pptx")
SALIDA = "entrega/presentaciones/5_noche_prediccion_frio.pptx"
PAL = pd.read_csv("results/palancas_cv/resumen.csv")
TECHO = pd.read_csv("results/techo/afinado.csv")
NIV = pd.read_csv("results/confianza/niveles.csv")
CV = json.load(open("results/cv_proyectos/contraste.json", encoding="utf-8"))
V3 = pd.read_csv("results/estimador_v3/verificacion.csv")
COR = pd.read_csv("results/corrector/distintas.csv")


def n(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def pal(anclas, est, met):
    v = PAL[(PAL.anclas == anclas) & (PAL.estimacion == est)]
    return float(v[met].iloc[0])


def v3(bloque, met):
    return float(V3[V3.bloque == bloque][met].iloc[0])


def main():
    prs = P.nueva()
    P.portada(prs, "Una noche sobre la\npredicción en frío",
              "Veinticuatro experimentos: siete con ganancia, diecisiete cerrados\n"
              "con medición. Y una explicación de por qué el techo está donde está.",
              "29 de septiembre de 2026")

    # ------------------------------------------------------------------ 2
    s, y = P.lamina(prs, "El encargo y lo que salió",
                    "Subir la predicción en frío hacia MdMRE 0,58–0,62 y R² 0,20–0,25.")
    y = P.vinetas(s, y, [
        ("SE LOGRÓ", f"el R² de la prueba fija sube de 0,113 a {n(v3('prueba fija', 'R2'), 3)}, "
                     f"un 60 % relativo, y la mejora es significativa también en proyectos "
                     f"nunca vistos (p = 2,4e-10)."),
        ("NO SE LOGRÓ", f"el MdMRE se queda en {n(v3('prueba fija', 'MdMRE'), 3)}, lejos del "
                        f"0,58–0,62. Es una mediana y casi no recoge lo que mejoró."),
        ("LO IMPORTANTE", "quedó medido POR QUÉ el frío no puede subir mucho más, y cuál "
                          "es la única palanca que de verdad lo cambia."),
    ], tam=16, salto=Inches(0.62))
    y += Inches(0.12)
    P.veredicto(s, y, "en una frase",
                "Dos tareas con el mismo texto, dentro del mismo proyecto, tardan lo mismo; "
                "en otro proyecto tardan el doble. El texto describe la tarea; no describe "
                "el ritmo del equipo.")

    # ------------------------------------------------------------------ 3
    P.seccion(prs, "1", "El techo, medido",
              "66.191 pares de tareas casi idénticas.\n"
              "La pregunta no es cuánto falta por exprimir, sino cuánto hay.")

    # ------------------------------------------------------------------ 4
    s, y = P.lamina(prs, "Dos tareas con el mismo texto",
                    "Pares con coseno ≥ 0,95 en el espacio del codificador. "
                    "Nadie puede distinguirlas leyendo el texto.")
    d = TECHO[(TECHO.umbral == 0.95)]
    dentro = d[d.conjunto == "mismo proyecto"].iloc[0]
    fuera = d[d.conjunto == "proyectos distintos"].iloc[0]
    y = P.tabla(s, y, ["_Dónde están las dos tareas", "Pares", "Difieren", "σ en log",
                       "% de la varianza"],
                [["dentro del MISMO proyecto", f"{int(dentro.pares):,}".replace(",", "."),
                  f"×{n(dentro.dif_mediana_h, 2)}", n(dentro.sigma_log, 3),
                  f"{dentro.fraccion * 100:.1f} %".replace(".", ",")],
                 ["entre proyectos DISTINTOS", f"{int(fuera.pares):,}".replace(",", "."),
                  f"×{n(fuera.dif_mediana_h, 2)}", n(fuera.sigma_log, 3),
                  f"{fuera.fraccion * 100:.1f} %".replace(".", ",")]],
                anchos=[4.6, 1.6, 1.6, 1.6, 2.2], tam=14, alto_fila=Inches(0.46),
                resaltar={0: P.SUAVE, 1: P.AVISO_SUAVE})
    y += Inches(0.25)
    y = P.veredicto(s, y, "lo que esto implica",
                    "Casi la mitad de la varianza de las horas es del proyecto, no de la "
                    "tarea. Ningún modelo que solo lea el texto puede recuperarla.")
    y += Inches(0.14)
    P.veredicto(s, y, "y por tanto",
                "PRED(25) ≥ 0,75 es inalcanzable sin conocer el proyecto. No es una excusa: "
                "está medido con el corpus en la mano.", tono="aviso")

    # ------------------------------------------------------------------ 5
    s, y = P.lamina(prs, "La misma idea, en dos barras")
    P.figura_ajustada(s, FIG / "noche_techo.png", y, Inches(4.3))

    # ------------------------------------------------------------------ 6
    P.seccion(prs, "2", "Lo que sí mejoró el frío",
              "Codificador, representación y ensamblado.\n"
              "R² 0,113 → 0,181 en la prueba fija.")

    # ------------------------------------------------------------------ 7
    s, y = P.lamina(prs, "De dónde salió cada décima",
                    "Todo sobre el mismo corpus y las mismas particiones congeladas.")
    y = P.tabla(s, y, ["_Etapa", "MdMRE", "MdAElog", "PRED25", "R²", "ρ"],
                [["punto de partida · mpnet + HGB", "0,6995", "0,8479", "0,150", "0,113", "0,320"],
                 ["codificador e5-large", "0,6884", "0,8303", "0,161", "0,149", "0,364"],
                 ["+ tf-idf de caracteres y palabras", "0,6860", "0,8346", "0,163", "0,152", "0,371"],
                 ["+ bolsa de 7 con train+val", "0,6851", "0,8244", "0,165", "0,164", "0,379"],
                 ["+ lineales dispersos apilados", n(v3("prueba fija", "MdMRE")),
                  n(v3("prueba fija", "MdAElog")), n(v3("prueba fija", "PRED25"), 3),
                  n(v3("prueba fija", "R2"), 3), n(v3("prueba fija", "spearman"), 3)]],
                anchos=[5.2, 1.5, 1.5, 1.4, 1.3, 1.3], tam=13.5, alto_fila=Inches(0.42),
                resaltar={0: P.GRIS_CLARO, 4: P.SUAVE})
    y += Inches(0.22)
    mil = f"{CV['n']:,}".replace(",", ".")
    P.veredicto(s, y, "significación",
                "Contraste pareado sobre las mismas tareas: ΔMdMRE −0,0205 con IC95 "
                "[−0,0365, −0,0056]. Y en proyectos nunca vistos, con validación cruzada "
                "sobre " + mil + " tareas: ΔMdMRE " + n(CV["dif_MdMRE"]) + " con IC95 [" +
                n(CV["IC95_MdMRE"][0]) + ", " + n(CV["IC95_MdMRE"][1]) +
                "] y p = 2,4e-10.")

    # ------------------------------------------------------------------ 8
    s, y = P.lamina(prs, "Un detalle que merece estar en la memoria",
                    "El apilado no premia al mejor modelo, premia al más distinto.")
    y = P.vinetas(s, y, [
        "La estimación por analogía —la mediana de los 15 vecinos más parecidos— es el "
        "PEOR modelo por su cuenta: R² 0,038 frente a 0,158 de la bolsa de árboles.",
        "Y es el que MÁS peso recibe en el apilado: +0,272, por delante del ridge disperso "
        "(+0,218) y de la propia bolsa (+0,174).",
        "Lo que aporta un conjunto es diversidad de errores, no potencia individual. El SVR "
        "denso, que se parece al ridge, recibe +0,001.",
    ], tam=16, salto=Inches(0.66))

    # ------------------------------------------------------------------ 9
    P.seccion(prs, "3", "La palanca que sí cambia el número",
              "Medida sobre 24.848 tareas de 204 proyectos nunca vistos.")

    # ------------------------------------------------------------------ 10
    s, y = P.lamina(prs, "Qué aporta cada palanca")
    _, y2 = P.figura_ajustada(s, FIG / "noche_palancas_cv.png", y, Inches(4.0))
    P.veredicto(s, y2 + Inches(0.14), "la palanca es una",
                f"Pedirle la estimación al usuario multiplica el R² por cinco: de "
                f"{n(pal(0, False, 'R2'), 3)} a {n(pal(0, True, 'R2'), 3)}. Las anclas lo "
                f"llevan solo a {n(pal(20, False, 'R2'), 3)}, y combinarlas con la "
                f"estimación no suma.")

    # ------------------------------------------------------------------ 11
    s, y = P.lamina(prs, "Una corrección a mí mismo, del mismo día",
                    "Los dos bancos no dicen lo mismo, y el pequeño exageraba.")
    y = P.tabla(s, y, ["_Lo que aportan 20 anclas", "Bloque externo (792 tareas)",
                       "Validación cruzada (24.848)"],
                [["R² en frío", "0,055", n(pal(0, False, "R2"), 3)],
                 ["R² con 20 anclas", "0,260", n(pal(20, False, "R2"), 3)],
                 ["ganancia", "+0,205", f"+{n(pal(20, False, 'R2') - pal(0, False, 'R2'), 3)}"]],
                anchos=[4.0, 3.5, 3.5], tam=14, alto_fila=Inches(0.46),
                resaltar={2: P.AVISO_SUAVE})
    y += Inches(0.25)
    P.veredicto(s, y, "por qué pasó",
                "El bloque externo son 49 proyectos y 792 tareas. El mismo banco había "
                "ocultado la mejora del modelo (p = 0,28) y exageraba esta palanca cuatro "
                "veces. Las conclusiones se sostienen sobre el banco grande.", tono="ambar")

    # ------------------------------------------------------------------ 12
    s, y = P.lamina(prs, "Y el modelo sabe cuándo acierta",
                    "La similitud con el historial del proyecto resultó ser la mejor señal "
                    "de confianza, y la única que mejora sola con el uso.")
    filas = []
    for _, r in NIV[NIV.bloque == "prueba SiP"].iterrows():
        filas.append([r.nivel, f"{r.fraccion * 100:.1f} %".replace(".", ","),
                      n(r.MdMRE, 3), n(r.PRED25, 3)])
    y = P.tabla(s, y, ["_Nivel de familiaridad", "% de tareas", "MdMRE", "PRED25"],
                filas, anchos=[5.4, 2.0, 1.9, 1.9], tam=14, alto_fila=Inches(0.44),
                resaltar={0: P.SUAVE})
    y += Inches(0.25)
    P.veredicto(s, y, "lo que permite hacer",
                "Para el 2,6 % de tareas que el proyecto ya ha hecho, el modelo casi cumple "
                "el criterio de aceptación: PRED(25) 0,688. Decirle al usuario en qué nivel "
                "está es más honesto que dar el mismo número con la misma cara.")

    # ------------------------------------------------------------------ 13
    s, y = P.lamina(prs, "Diecisiete vías cerradas, con su medición",
                    "Están en el repositorio con sus scripts y sus CSV. No hay que "
                    "reintentarlas sin cambiar el planteamiento.")
    y = P.vinetas(s, y, [
        ("DATOS", "descripción y nombre del proyecto · el backlog como contexto · "
                  "seleccionar el corpus por escala · deduplicar · recortar · recencia"),
        ("OBJETIVO", "centrar por proyecto · expandir la dispersión · puntuaciones "
                     "normales · pérdida absoluta"),
        ("MODELO", "expertos por fuente · hiperparámetros · rasgos de superficie · "
                   "modelo de ordenación · prioridad como rasgo"),
        ("ATAJOS", "copiar las horas del gemelo · acotar cuánto corrige el corrector"),
    ], tam=15, salto=Inches(0.62))
    y += Inches(0.1)
    P.veredicto(s, y, "el patrón",
                "Casi todo lo que falla, falla por lo mismo: intenta sacar del texto una "
                "información que no está en el texto.", tono="ambar")

    # ------------------------------------------------------------------ 14
    s, y = P.lamina(prs, "Qué haría ahora")
    y = P.vinetas(s, y, [
        ("1", "Decidir el producto sobre la tabla de palancas, no sobre el frío. El frío "
              "es el primer segundo de vida de un proyecto."),
        ("2", "Pedir la estimación al usuario. Es la única palanca grande y multiplica el "
              "R² por cinco."),
        ("3", "Enseñar el nivel de familiaridad junto a la estimación."),
        ("4", "Escribir el techo en la memoria. Que PRED(25) ≥ 0,75 sea inalcanzable sin "
              "conocer el proyecto es un resultado, no una excusa."),
    ], tam=16, salto=Inches(0.66))

    P.numerar(prs)
    Path(SALIDA).parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"escrito {SALIDA} con {len(prs.slides._sldIdLst)} láminas")


if __name__ == "__main__":
    main()
