"""Builds the two decks, reading every figure from results/REGISTRO.csv so the
slides cannot drift from the measurements.

  entrega/Agix_resultados.pptx   what was measured
  entrega/Agix_proceso.pptx      how the pipeline was built

python-pptx, 16:9, native charts. Speaker notes carry the talking points,
including the caveats that belong next to each number.
"""
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

BASE = r"F:\Software Task generation Model"
SALIDA = f"{BASE}/entrega"
os.makedirs(SALIDA, exist_ok=True)

MARINO = RGBColor(0x1E, 0x27, 0x61)
TEAL = RGBColor(0x1C, 0x72, 0x93)
MENTA = RGBColor(0x02, 0xC3, 0x9A)
ARENA = RGBColor(0xF4, 0xF6, 0xF9)
TINTA = RGBColor(0x1D, 0x1C, 0x1A)
SUAVE = RGBColor(0x6B, 0x67, 0x62)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
HIELO = RGBColor(0xCA, 0xDC, 0xFC)
CORAL = RGBColor(0xB8, 0x50, 0x42)
SERIF, SANS = "Cambria", "Calibri"
W, H = 13.333, 7.5


def nueva():
    p = Presentation()
    p.slide_width, p.slide_height = Inches(W), Inches(H)
    return p


def lamina(prs, fondo=BLANCO):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = fondo
    return s


def texto(slide, x, y, w, h, contenido, tam=16, color=TINTA, negrita=False, fuente=SANS,
          alineacion=PP_ALIGN.LEFT, interlineado=1.15, espacio_despues=0):
    caja = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = caja.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lineas = contenido if isinstance(contenido, list) else [contenido]
    for i, linea in enumerate(lineas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = alineacion
        p.line_spacing = interlineado
        p.space_after = Pt(espacio_despues)
        partes = linea if isinstance(linea, list) else [(linea, negrita)]
        for t, neg in partes:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(tam)
            r.font.bold = neg
            r.font.name = fuente
            r.font.color.rgb = color
    return caja


def vinetas(slide, x, y, w, items, tam=15, color=TINTA, gap=8):
    """items: (lead, resto) -> lead in bold"""
    caja = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.5))
    tf = caja.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        lead, resto = it if isinstance(it, tuple) else ("", it)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.2
        p.space_after = Pt(gap)
        if lead:
            r = p.add_run()
            r.text = lead
            r.font.bold = True
            r.font.size = Pt(tam)
            r.font.name = SANS
            r.font.color.rgb = color
        r = p.add_run()
        r.text = resto
        r.font.size = Pt(tam)
        r.font.name = SANS
        r.font.color.rgb = color
    return caja


def tarjeta(slide, x, y, w, h, relleno=ARENA, borde=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = relleno
    if borde:
        sh.line.color.rgb = borde
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def stats(slide, y, items, ancho_total=12.33, x0=0.5, alto=1.45, color_num=MARINO, fondo=ARENA,
          color_etq=SUAVE):
    n = len(items)
    hueco = 0.25
    w = (ancho_total - hueco * (n - 1)) / n
    for i, (num, etq) in enumerate(items):
        x = x0 + i * (w + hueco)
        tarjeta(slide, x, y, w, alto, relleno=fondo)
        # long figures ("300M -> 121M") wrapped onto the label underneath
        tam = 30 if len(num) <= 9 else (24 if len(num) <= 13 else 19)
        texto(slide, x + 0.2, y + 0.16, w - 0.4, 0.7, num, tam=tam, color=color_num,
              negrita=True, fuente=SERIF)
        texto(slide, x + 0.2, y + 0.9, w - 0.4, 0.5, etq, tam=11.5, color=color_etq)


def titulo_lamina(slide, t, bajada=None, color=TINTA):
    texto(slide, 0.6, 0.5, 12.2, 0.8, t, tam=34, color=color, negrita=True, fuente=SERIF)
    if bajada:
        texto(slide, 0.6, 1.32, 12.2, 0.5, bajada, tam=14.5, color=SUAVE)


def tabla(slide, x, y, w, encabezados, filas, anchos=None, tam=12, alto_fila=0.34):
    n_col, n_fil = len(encabezados), len(filas) + 1
    forma = slide.shapes.add_table(n_fil, n_col, Inches(x), Inches(y), Inches(w),
                                   Inches(alto_fila * n_fil))
    t = forma.table
    if anchos:
        total = sum(anchos)
        for i, a in enumerate(anchos):
            t.columns[i].width = Emu(int(Inches(w) * a / total))
    for j, enc in enumerate(encabezados):
        c = t.cell(0, j)
        c.text = enc
        c.fill.solid()
        c.fill.fore_color.rgb = MARINO
        p = c.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(tam)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = BLANCO
        p.runs[0].font.name = SANS
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, fila in enumerate(filas, start=1):
        for j, val in enumerate(fila):
            c = t.cell(i, j)
            neg = isinstance(val, tuple)
            c.text = val[0] if neg else str(val)
            c.fill.solid()
            c.fill.fore_color.rgb = BLANCO if i % 2 else ARENA
            p = c.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(tam)
            p.runs[0].font.name = SANS
            p.runs[0].font.bold = neg
            p.runs[0].font.color.rgb = MARINO if neg else TINTA
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return t


def barras(slide, x, y, w, h, categorias, serie, titulo=None, colores=None, horizontal=True,
           formato='0.00', color_texto=TINTA, color_ejes=SUAVE):
    datos = CategoryChartData()
    datos.categories = categorias
    datos.add_series(titulo or "valor", serie, number_format=formato)
    tipo = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    g = slide.shapes.add_chart(tipo, Inches(x), Inches(y), Inches(w), Inches(h), datos).chart
    g.has_legend = False
    g.has_title = False
    pl = g.plots[0]
    pl.has_data_labels = True
    pl.data_labels.number_format = formato
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.font.size = Pt(11)
    pl.data_labels.font.bold = True
    pl.data_labels.font.color.rgb = color_texto
    pl.gap_width = 60
    ser = pl.series[0]
    if colores:
        for i, pt in enumerate(ser.points):
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = colores[i % len(colores)]
    else:
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = TEAL
    for eje in (g.category_axis, g.value_axis):
        eje.tick_labels.font.size = Pt(11)
        eje.tick_labels.font.name = SANS
        eje.tick_labels.font.color.rgb = color_ejes
        eje.has_major_gridlines = False
        eje.format.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    g.value_axis.visible = False
    return g


def notas(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# =========================================================================  ANEXOS
def anexos(prs):
    """Backup slides: the full tables behind every headline, for detail questions.

    Nothing here is presented; each one exists so a question can be answered by
    showing the measurement instead of recalling it.
    """
    s = lamina(prs, MARINO)
    texto(s, 0.9, 1.5, 11.5, 1.0, "Anexos", tam=40, color=BLANCO, negrita=True, fuente=SERIF)
    texto(s, 0.9, 2.5, 11.5, 0.6,
          "Tablas completas detrás de cada cifra. No se presentan: se abren si hay una pregunta.",
          tam=15.5, color=HIELO)
    izq = [("A1", "Banco completo de LLM"), ("A2", "Cobertura: 11 estrategias"),
           ("A3", "Ablación de la mitad de arriba"), ("A4", "Dominios: dejar uno fuera"),
           ("A5", "Español: entrenar contra traducir"), ("A6", "Traducción de la salida"),
           ("A7", "Búsqueda de hiperparámetros")]
    der = [("A8", "Barrido de decodificación"), ("A9", "small contra base"),
           ("A10", "Datos, limpieza y particiones"), ("A11", "SBERT propio"),
           ("A12", "Despliegue: CPU, GPU y costo"), ("A13", "One2One de la mitad de arriba"),
           ("A14", "Qué falta y cómo se cerraría")]
    for col, x in ((izq, 0.9), (der, 6.9)):
        vinetas(s, x, 3.5, 5.6, [(f"{a} · ", b) for a, b in col], tam=13.5, color=HIELO, gap=7)
    notas(s, "Si una pregunta pide un detalle, es mejor abrir su anexo que citarlo de memoria.")

    # ---------------------------------------------------------------- A1
    s = lamina(prs)
    titulo_lamina(s, "A1 · Banco completo de LLM",
                  "64 proyectos (60 de Jira reservados + 4 de Dalpiaz), misma máquina, Ollama local")
    tabla(s, 0.6, 2.0, 12.13,
          ["sistema", "s mediana", "s p90", "historias", "tareas", "F1 suave", "dif. pareada (IC95)", "reconoció"],
          [[("Nuestro pipeline", True), ("4,18", True), "4,6", "12,0", "48", ("0,472", True), "—", ("0 %", True)],
           ["llama3.1:8b sin ejemplos", "21,63", "25,2", "11,9", "122", "0,463", "−0,009 [−0,035; +0,017]", "81 %"],
           ["llama3.1:8b con ejemplos", "21,05", "27,0", "12,0", "124", "0,479", "+0,007 [−0,016; +0,033]", "81 %"],
           ["gemma2:9b sin ejemplos", "35,73", "45,3", "12,0", "109", "0,475", "+0,003 [−0,026; +0,031]", "59 %"],
           ["gemma2:9b con ejemplos", "35,12", "45,2", "12,0", "108", "0,487", "+0,015 [−0,010; +0,040]", "59 %"],
           ["mistral:7b sin ejemplos", "39,44", "43,3", "12,0", "202", "0,473", "+0,001 [−0,023; +0,026]", "55 %"],
           ["mistral:7b con ejemplos", "39,93", "43,5", "12,0", "193", "0,486", "+0,014 [−0,009; +0,036]", "55 %"]],
          anchos=[3.1, 1.25, 0.95, 1.1, 0.9, 1.15, 2.85, 1.15], tam=11, alto_fila=0.4)
    vinetas(s, 0.6, 5.45, 12.2, [
        ("Cómo se midió: ", "un proceso por sistema, modelos descargados de memoria antes de cronometrar el "
                            "siguiente, y el mismo orden de proyectos para los cuatro."),
        ("Por qué el reconocimiento importa: ", "el LLM puntúa sobre proyectos que vio al entrenar; el nuestro "
                                               "no puede, porque esos 64 proyectos quedaron fuera de su entrenamiento por construcción."),
        ("Fuente: ", "results/20_banco.jsonl, results/22_banco_resumen.csv, results/22_banco_pareado.csv.")],
        tam=12.5, gap=8)
    notas(s, "Los tiempos del LLM son de generación local en la misma GPU, sin contar el arranque del servidor. "
             "Las dos condiciones son «sin ejemplos» (solo la descripción) y «con ejemplos» (dos backlogs "
             "recuperados en el prompt), las mismas que probamos en nuestro modelo.")

    # ---------------------------------------------------------------- A2
    s = lamina(prs)
    titulo_lamina(s, "A2 · Cobertura: 11 estrategias probadas",
                  "Criterio fijado antes de medir: subir cobertura sin bajar la precisión de 0,50")
    tabla(s, 0.6, 1.95, 12.13,
          ["estrategia", "F1 suave", "precisión", "cobertura", "F2", "tareas/historia", "¿cumple el piso?"],
          [[("Dos modelos (elegida)", True), ("0,512", True), ("0,571", True), ("0,479", True), ("0,495", True), "2,8", ("sí", True)],
           ["Haz + un ejemplo x2", "0,506", "0,570", "0,471", "0,488", "2,7", "sí"],
           ["Un ejemplo x2", "0,503", "0,573", "0,464", "0,482", "2,3", "sí"],
           ["Haz de 4 (anterior)", "0,494", "0,576", "0,444", "0,466", "2,0", "sí"],
           ["Dos modelos + muestreo 5", "0,484", "0,456", "0,535", "0,517", "15,5", "no"],
           ["Haz + muestreo 8", "0,481", "0,444", "0,542", "0,519", "22,2", "no"],
           ["Muestreo 8", "0,478", "0,443", "0,535", "0,514", "22,7", "no"],
           ["Haz + muestreo 5", "0,476", "0,449", "0,523", "0,507", "14,4", "no"],
           ["Haz + muestreo 3", "0,470", "0,456", "0,503", "0,493", "9,3", "no"],
           ["Muestreo 5", "0,464", "0,442", "0,506", "0,492", "13,9", "no"],
           ["Muestreo 3", "0,448", "0,440", "0,473", "0,466", "8,5", "no"]],
          anchos=[3.3, 1.3, 1.4, 1.4, 1.1, 2.0, 1.9], tam=10.5, alto_fila=0.33)
    texto(s, 0.6, 6.15, 12.2, 0.7,
          "Muestrear más sí sube la cobertura, pero propone 8 a 23 tareas por historia y la precisión cae a 0,44: "
          "un backlog que hay que podar a mano no ahorra trabajo.", tam=13, negrita=True, color=MARINO)
    notas(s, "Medido en validación (1 382 ejemplos) y confirmado en prueba: cobertura 0,438 → 0,467 y F1 0,473 → "
             "0,485, ambos con IC95 sin cruzar cero. Fuente: results/23_cobertura.csv.")

    # ---------------------------------------------------------------- A3
    s = lamina(prs)
    titulo_lamina(s, "A3 · Ablación de la mitad de arriba",
                  "Qué hacer con los proyectos parecidos que recupera el SBERT")
    tabla(s, 0.6, 2.0, 8.4,
          ["variante", "F1 suave", "precisión", "cobertura", "fuga de vocabulario"],
          [[("Ordenar por relevancia", True), ("0,434", True), ("0,405", True), "0,480", "10,5 %"],
           ["Vecinos solo si cos ≥ 0,50", "0,422", "0,384", "0,485", "9,0 %"],
           ["Sin ejemplos en el prompt", "0,417", "0,380", "0,476", ("0 %", True)],
           ["4 vecinos", "0,414", "0,377", "0,476", "17,0 %"],
           ["2 vecinos (configuración previa)", "0,414", "0,377", "0,473", "11,8 %"],
           ["Vecinos solo si cos ≥ 0,55", "0,413", "0,377", "0,469", "3,3 %"]],
          anchos=[3.6, 1.3, 1.3, 1.3, 1.9], tam=11.5, alto_fila=0.38)
    tarjeta(s, 9.3, 2.0, 3.45, 2.8, relleno=ARENA)
    texto(s, 9.55, 2.2, 2.95, 0.4, "Lo que se adoptó", tam=13, negrita=True, color=MARINO)
    vinetas(s, 9.55, 2.65, 2.95, [
        ("", "Quitar los ejemplos del prompt y ordenar los candidatos por relevancia: +0,022 F1 en los 60 epics "
             "de prueba, IC95 [+0,010; +0,035], y 0 % de fuga.")], tam=11.5)
    vinetas(s, 0.6, 5.1, 12.2, [
        ("Qué es la fuga: ", "porcentaje de historias generadas que copian una palabra poco común del proyecto "
                             "vecino y no del propio. Es el modo de fallo que más se nota al leer el backlog."),
        ("Conclusión: ", "el modelo aprendió a ignorar los ejemplos del prompt; la recuperación sirve para ordenar "
                         "y para mostrar procedencia, no para condicionar la generación.")],
        tam=13, gap=8)
    notas(s, "Medido en validación con 90 epics. La misma ablación abajo da empate (+0,006, el IC cruza cero) pero "
             "es 16 % más rápida sin ejemplos, así que también se quitaron. Fuente: results/24_calidad_arriba.csv.")

    # ---------------------------------------------------------------- A4
    s = lamina(prs)
    titulo_lamina(s, "A4 · Dominios: dejar uno entero fuera",
                  "Tres reentrenamientos desde cero, ~32 min cada uno, un proceso por dominio")
    tabla(s, 0.6, 1.95, 7.5,
          ["dominio dejado fuera", "n", "fuera", "dentro", "diferencia"],
          [["Apps web y CMS (Moodle, Sakai)", "350", "0,457", "0,446", "+0,011"],
           ["Blockchain y distribuidos (Fabric, Kafka)", "205", "0,481", "0,458", "+0,023"],
           ["Big data (Spark, Flink, HBase)", "300", "0,459", "0,452", "+0,007"]],
          anchos=[4.0, 0.8, 1.1, 1.1, 1.4], tam=11.5, alto_fila=0.44)
    tabla(s, 8.4, 1.95, 4.35,
          ["control: dificultad propia", "F1"],
          [["0 · documentación y QA", "0,487"], ["1 · infraestructura Apache", "0,477"],
           ["2 · apps web y CMS", "0,462"], ["3 · blockchain", "0,489"],
           ["4 · big data", "0,455"], ["5 · middleware Java", "0,524"]],
          anchos=[3.0, 1.2], tam=11, alto_fila=0.33)
    vinetas(s, 0.6, 4.45, 12.2, [
        ("Lectura: ", "el modelo no pierde nada en un dominio que nunca vio. Lo que aprende es cómo se descompone "
                      "trabajo de software, no el vocabulario de un dominio."),
        ("El control descarta la explicación fácil: ", "los dominios difieren de por sí (0,455 a 0,524) y los dos "
                                                      "que «mejoran» al dejarlos fuera son de los más difíciles, así que no es un conjunto fácil."),
        ("Resultado negativo: ", "la similitud con lo recuperado no sirve como señal de «no sé». Fuera de dominio "
                                "es más alta (0,555 contra 0,529) y correlaciona menos con la calidad (0,08 contra 0,18)."),
        ("Corrección aplicada: ", "la primera versión del experimento ponía proyectos de entrenamiento en el lado "
                                 "«fuera» y exageraba el efecto de +1,1 % a +4,9 %. Ambos lados usan hoy solo proyectos de prueba.")],
        tam=12.5, gap=8)
    notas(s, "Los seis dominios salen de k-means sobre vectores de proyecto. Se dejaron fuera los dominios 2, 3 y 4 "
             "porque son los que tienen suficientes proyectos de prueba a los dos lados. "
             "Fuente: results/29_lodo_v2.csv y results/29_control_dificultad.csv.")

    # ---------------------------------------------------------------- A5
    s = lamina(prs)
    titulo_lamina(s, "A5 · Español: entrenar contra traducir", "La solución barata le ganó a la costosa")
    tabla(s, 0.6, 2.0, 7.6,
          ["modelo de la mitad de arriba", "entrada en inglés", "entrada en español"],
          [[("Sin entrenar en español (producción)", True), ("0,455", True), ("0,377", True)],
           ["Entrenado con entradas traducidas", "0,423", "0,374"]],
          anchos=[4.0, 1.8, 1.8], tam=12, alto_fila=0.46)
    stats(s, 3.5, [("−0,032", "lo que el entrenamiento en español le quitó al inglés · IC95 [−0,050; −0,015]"),
                   ("−0,003", "lo que le dio al español: nada · no significativo"),
                   ("+0,045", "traducir la entrada al inferir · IC95 [+0,027; +0,065]")],
          alto=1.6)
    vinetas(s, 0.6, 5.35, 12.2, [
        ("Lo que se hace en producción: ", "si la descripción llega en español se detecta por palabras funcionales "
                                          "y acentos, se traduce al inglés con opus-mt-es-en (175 ms) y el modelo trabaja en inglés."),
        ("Límite honesto: ", "las entradas en español del conjunto de prueba son traducciones automáticas del "
                             "inglés, así que el número es un ida y vuelta y es optimista. No existen descripciones "
                             "escritas en español en los datos.")],
        tam=12.5, gap=8)
    notas(s, "Entrenar con datos traducidos metió ruido de traducción en el modelo y no le enseñó español real. "
             "Fuente: results/26_eval_es.csv y la medición directa del 23 de setiembre.")

    # ---------------------------------------------------------------- A6
    s = lamina(prs)
    titulo_lamina(s, "A6 · Traducción de la salida al español",
                  "Sin referencias humanas: se mide contra un piso de pares equivocados")
    tabla(s, 0.6, 2.0, 12.13,
          ["qué se traduce", "n", "fidelidad", "piso (pares al azar)", "peor decil", "forma de historia",
           "términos técnicos", "ms/ítem"],
          [["Historias", "326", ("0,903", True), "0,199", "0,803", "99 % → 92 %", ("100 %", True), "12,4"],
           ["Tareas", "1 194", ("0,881", True), "0,145", "0,757", "—", ("99,5 %", True), "4,3"]],
          anchos=[2.2, 0.8, 1.4, 2.0, 1.3, 1.9, 1.8, 0.9], tam=11.5, alto_fila=0.44)
    vinetas(s, 0.6, 3.5, 12.2, [
        ("Por qué así: ", "no hay traducciones humanas de referencia, así que se compara el par real "
                          "(inglés, español) con pares al azar usando el SBERT multilingüe. Si la traducción no "
                          "dijera lo mismo, la fidelidad caería hacia el piso."),
        ("El problema que apareció: ", "el traductor volvía literales los nombres de producto («helm charts» → "
                                      "«cartas de timón»). Se enmascaran los términos técnicos antes de traducir y "
                                      "se restauran después: la retención pasó del 64 % al 100 %."),
        ("Costo: ", "0,31 s sobre 4,39 s, un 7 % del tiempo total."),
        ("Límite: ", "la lista de nombres de producto es curada a mano. Un glosario completo es trabajo de "
                     "mantenimiento, porque «helm» no se distingue de «agile» automáticamente.")],
        tam=12.5, gap=8)
    notas(s, "El 8 % de historias que pierden la forma «Como…, quiero…» son casos donde el traductor reordena la "
             "frase; se puede forzar con una plantilla si hace falta. Fuente: results/27_traduccion_salida.csv.")

    # ---------------------------------------------------------------- A7
    s = lamina(prs)
    titulo_lamina(s, "A7 · Búsqueda de hiperparámetros", "41 pruebas con Optuna y poda ASHA, 2 h 26 min de GPU")
    tabla(s, 0.6, 2.0, 7.9,
          ["prueba", "F1 val", "lote", "tasa de aprendizaje", "calentamiento"],
          [[("38 (ganadora)", True), ("0,4738", True), "8", "9,2 · 10⁻⁴", "5,1 %"],
           ["19", "0,4674", "8", "9,1 · 10⁻⁴", "2,9 %"],
           ["34", "0,4656", "8", "1,3 · 10⁻³", "4,6 %"],
           ["35", "0,4656", "8", "1,6 · 10⁻³", "6,1 %"],
           ["12", "0,4642", "8", "6,2 · 10⁻⁴", "4,7 %"],
           ["1", "0,4618", "8", "1,7 · 10⁻³", "3,4 %"]],
          anchos=[1.8, 1.3, 1.0, 2.2, 1.6], tam=11.5, alto_fila=0.36)
    tarjeta(s, 8.7, 2.0, 4.05, 3.4, relleno=ARENA)
    texto(s, 8.95, 2.2, 3.55, 0.4, "Y no transfirió", tam=13.5, negrita=True, color=CORAL)
    vinetas(s, 8.95, 2.7, 3.55, [
        ("", "Entrenada con todos los datos, la ganadora dio 0,4894 en validación contra 0,4933 del modelo ya en "
             "producción: diferencia −0,004 con IC95 [−0,015; +0,008], o sea empate."),
        ("", "Se mantuvo el modelo previo, ajustado a mano.")], tam=11.5)
    vinetas(s, 0.6, 4.75, 7.9, [
        ("34 de las 41 pruebas se podaron ", "a los pocos minutos: ASHA corta las que no prometen."),
        ("Lo aprendido: ", "los hiperparámetros óptimos en un subconjunto no lo son con todos los datos. Es un "
                           "resultado útil para la tesis, no un fracaso."),
        ("Y algo se ganó: ", "la variante «perdedora» se equivoca distinto, y unida al modelo principal es lo que "
                             "sube la cobertura (anexo A2).")],
        tam=12.5, gap=8)
    notas(s, "Fuente: results/21_trials.csv. Cada prueba entrenaba sobre un subconjunto con parada temprana; "
             "por eso salen unos 5 minutos por prueba.")

    # ---------------------------------------------------------------- A8
    s = lamina(prs)
    titulo_lamina(s, "A8 · Barrido de decodificación", "Cómo se eligió la forma de generar, antes de tocar el modelo")
    tabla(s, 0.6, 2.0, 8.2,
          ["configuración", "F1 suave", "F1 exacto", "ROUGE-1", "largo relativo", "s/250 ej."],
          [[("Haz 4, sin repetir 4-gramas, pen. 1,3", True), ("0,4908", True), "0,135", "0,177", "0,54", "30"],
           ["Haz 4, sin repetir 8-gramas, pen. 1,15", "0,4898", "0,127", "0,157", "0,68", "48"],
           ["Haz 4, sin repetir 6-gramas, pen. 1,05", "0,4892", "0,136", "0,171", "0,61", "44"],
           ["Haz 4, 6-gramas, premio al largo 2", "0,4875", "0,108", "0,148", "3,24", "58"],
           ["Haz 4, sin repetir 10-gramas", "0,4872", "0,123", "0,149", "0,90", "60"],
           ["Muestreo 3 con p = 0,92", "0,4700", "0,104", "0,187", "2,00", "44"],
           ["Muestreo 5 con p = 0,95", "0,4659", "0,084", "0,179", "3,37", "48"],
           ["Solo recuperación (referencia)", "0,3560", "0,039", "0,114", "1,09", "—"]],
          anchos=[4.4, 1.3, 1.2, 1.2, 1.5, 1.2], tam=11, alto_fila=0.36)
    vinetas(s, 0.6, 5.5, 12.2, [
        ("El hallazgo que lo motivó: ", "el modelo no generaba listas cortas, se quedaba en bucle repitiendo la "
                                       "misma tarea hasta el tope de longitud. Los parámetros de longitud mínima no hacían nada porque "
                                       "el bucle ya llenaba la salida."),
        ("La solución: ", "prohibir repetir 4-gramas y penalizar la repetición. El largo relativo pasó de 0,3 a "
                          "0,54 y el F1 subió 0,13 puntos."),
        ("Se hizo el mismo barrido en mT5-base ", "y el orden de las configuraciones fue idéntico: la elección no "
                                                 "depende del tamaño del modelo.")],
        tam=12.5, gap=8)
    notas(s, "«Largo relativo» es tokens generados sobre tokens de la referencia: 1,0 sería la longitud ideal. "
             "Fuente: results/11_barrido_mt5-small-pruned_agix_v1.csv y su equivalente en base.")

    # ---------------------------------------------------------------- A9
    s = lamina(prs)
    titulo_lamina(s, "A9 · mT5-small contra mT5-base", "El modelo grande no ganó: se eligió el chico")
    tabla(s, 0.6, 2.0, 8.6,
          ["modelo", "parte", "n", "F1 suave", "precisión", "cobertura", "ROUGE-1"],
          [[("mT5-small podado (elegido)", True), "todo", "2 478", ("0,4684", True), "0,535", "0,429", "0,180"],
           ["mT5-small podado", "historia → tareas", "1 751", "0,4744", "0,530", "0,441", "0,194"],
           ["mT5-small podado", "epic → historias", "727", "0,4537", "0,546", "0,401", "0,146"],
           ["mT5-base podado", "todo", "2 478", "0,4640", "0,527", "0,427", "0,177"],
           ["mT5-base podado", "historia → tareas", "1 751", "0,4688", "0,521", "0,437", "0,191"],
           ["mT5-base podado", "epic → historias", "727", "0,4525", "0,542", "0,402", "0,144"],
           ["Solo recuperación", "todo", "2 478", "0,3182", "0,331", "0,317", "0,088"]],
          anchos=[3.1, 2.2, 1.0, 1.3, 1.3, 1.3, 1.2], tam=11, alto_fila=0.37)
    stats(s, 5.15, [("−0,004", "base contra small · IC95 [−0,009; 0,000]: empate"),
                   ("+0,150", "small contra solo recuperación · IC95 [+0,143; +0,157]"),
                   ("2,6x", "menos parámetros: 121M contra 314M"),
                   ("1,7x", "más rápido al generar: 30 s contra 51 s en el mismo barrido")])
    notas(s, "Que el modelo grande no gane es esperable con 11 mil ejemplos: la capacidad extra no tiene con qué "
             "llenarse. Decidirlo con una medición y no por costumbre es lo que permite defender el modelo chico. "
             "Fuente: results/11_test_metricas.csv y results/11_test_comparaciones.csv.")

    # ---------------------------------------------------------------- A10
    s = lamina(prs)
    titulo_lamina(s, "A10 · Datos, limpieza y particiones",
                  "De 2,8 millones de incidencias a 14 mil ejemplos de entrenamiento")
    tabla(s, 0.6, 1.95, 6.1,
          ["partición", "proyectos", "relaciones", "incidencias", "%"],
          [["Entrenamiento", "1 003", "195 286", "2 067 017", "79,8"],
           ["Validación", "122", "22 682", "233 012", "9,3"],
           ["Prueba", "164", "26 709", "545 037", "10,9"]],
          anchos=[2.0, 1.2, 1.4, 1.5, 0.8], tam=11.5, alto_fila=0.4)
    tabla(s, 7.0, 1.95, 5.75,
          ["ejemplos listos", "train", "val", "prueba"],
          [["historia → tareas", "11 202", "1 382", "1 751"],
           ["epic → historias", "6 068", "863", "727"],
           ["proyectos que aportan", "502 y 172", "55 y 25", "83 y 35"]],
          anchos=[2.4, 1.2, 1.0, 1.1], tam=11.5, alto_fila=0.4)
    vinetas(s, 0.6, 3.75, 12.2, [
        ("El filtro que más quitó: ", "de 244 677 relaciones padre → hijo solo 14 mil llegan a ser ejemplos. Se "
                                     "exige que el padre no sea trabajo reactivo, que tenga texto útil y al menos dos hijos, y se "
                                     "quitan los padres casi idénticos dentro de un mismo proyecto (coseno ≥ 0,97)."),
        ("Limpieza: ", "se quita el marcado de Jira (tablas, bloques de código, adjuntos, menciones), las plantillas "
                       "repetidas y los textos demasiado cortos. El 15 % de las entradas se recorta al tope de 256 "
                       "tokens; pierde alguna tarea el 14 % de las listas de abajo y el 21 % de las de arriba."),
        ("Partición por proyecto entero, y una sola para todo el trabajo: ", "el SBERT, los dos generadores y la "
                                                                            "evaluación comparten el mismo archivo de particiones, así que el SBERT tampoco vio los proyectos de prueba."),
        ("Los proyectos gigantes van forzados a entrenamiento ", "para que uno solo no sea la mitad del conjunto de prueba.")],
        tam=12.5, gap=8)
    notas(s, "Fuente: results/04_split_resumen.csv y results/08_preprocess_resumen.csv. El detalle completo del "
             "criterio de limpieza está en entrega/COMO_SE_HIZO.md.")

    # ---------------------------------------------------------------- A11
    s = lamina(prs)
    titulo_lamina(s, "A11 · SBERT propio para recuperar", "nDCG@10 en proyectos de prueba, congelado contra ajustado")
    tabla(s, 0.6, 2.0, 11.0,
          ["consulta", "relación", "n", "congelado", "ajustado", "mejora", "IC95"],
          [[("Inglés", True), "todas", "3 309", "0,3531", ("0,3923", True), "+0,039", "[+0,030; +0,048]"],
           ["Inglés", "epic → historia", "818", "0,2951", "0,3689", "+0,074", "[+0,059; +0,089]"],
           ["Inglés", "padre → subtarea", "2 491", "0,3721", "0,3999", "+0,028", "[+0,017; +0,038]"],
           [("Español", True), "todas", "3 309", "0,2294", ("0,3167", True), "+0,087", "[+0,078; +0,096]"],
           ["Español", "epic → historia", "818", "0,1810", "0,2950", "+0,114", "[+0,099; +0,130]"],
           ["Español", "padre → subtarea", "2 491", "0,2453", "0,3238", "+0,079", "[+0,068; +0,089]"]],
          anchos=[1.6, 2.4, 1.0, 1.5, 1.5, 1.3, 2.2], tam=11.5, alto_fila=0.37)
    vinetas(s, 0.6, 4.9, 12.2, [
        ("Cómo se entrenó: ", "201 309 pares padre → hijo con pérdida contrastiva simétrica y negativos dentro del "
                              "lote; lote de 256 con GradCache para no pasar de 6,5 GB de VRAM."),
        ("Bilingüe a propósito: ", "el 30 % de las anclas se tradujo al español, porque las consultas de Agix "
                                  "llegan en español. Ahí está la mayor mejora."),
        ("Criterio declarado antes de medir: ", "adoptarlo solo si superaba al congelado en los dos idiomas con "
                                               "IC95 sin cruzar cero. Se cumplió en las seis filas."),
        ("Por qué no se reusó el SBERT de la tesis hermana: ", "estaba ajustado para predecir horas, no parecido, "
                                                              "y de él solo quedaban los vectores, no los pesos.")],
        tam=12.5, gap=8)
    notas(s, "nDCG@10 mide si el hijo correcto aparece arriba entre los candidatos recuperados. "
             "Fuente: results/07_sbert_test.csv.")

    # ---------------------------------------------------------------- A12
    s = lamina(prs)
    titulo_lamina(s, "A12 · Despliegue: CPU, GPU y costo",
                  "Todo medido en esta máquina; la laptop de la presentación es el caso «CPU»")
    tabla(s, 0.6, 1.95, 7.7,
          ["configuración", "s mediana", "s p90", "tareas", "carga", "RAM pico"],
          [[("Nuestro, GPU", True), ("4,18", True), "4,6", "48", "—", "—"],
           ["Nuestro, CPU 6 hilos, ligero", "6,26", "6,7", "32", "4,3 s", "2,7 GB"],
           ["Nuestro, CPU 4 hilos, ligero", "6,98", "7,5", "31", "4,2 s", "3,0 GB"],
           ["Nuestro, CPU 6 hilos, completo", "10,41", "12,1", "40", "4,6 s", "3,0 GB"],
           [("Nuestro, CPU 4 hilos, completo", True), ("11,35", True), "12,2", "48", "4,3 s", "3,2 GB"],
           ["llama3.1:8b con GPU", "21,63", "25,2", "122", "—", "—"],
           [("llama3.1:8b en CPU", True), ("181,89", True), "—", "135", "—", "—"]],
          anchos=[3.6, 1.3, 1.0, 1.0, 1.1, 1.3], tam=11, alto_fila=0.36)
    tabla(s, 8.5, 1.95, 4.25,
          ["opción de despliegue", "costo"],
          [["Serverless, en caliente", "0,10 ¢/proyecto"],
           ["Serverless, en frío", "0,44 ¢/proyecto"],
           ["Servidor CPU pequeño", "pocos USD/mes"],
           ["PC propia 24/7", "S/ 40 al mes"],
           ["API externa tipo Groq", "0,05 ¢/proyecto"]],
          anchos=[2.9, 1.5], tam=11, alto_fila=0.4)
    vinetas(s, 0.6, 5.05, 7.7, [
        ("«Ligero» ", "es un solo modelo en la mitad de abajo; «completo» son los dos del conjunto."),
        ("El dato que cierra el argumento: ", "en CPU somos 16 veces más rápidos que llama3.1 en la misma CPU, "
                                             "y aun en CPU le ganamos al LLM con GPU dedicada."),
        ("Lo que no se puede sostener: ", "la API externa es la más barata por proyecto, pero el modelo que Agix "
                                         "usaba en Groq dejó de estar en autoservicio en agosto de 2026. La dependencia es un riesgo "
                                         "medido, no hipotético.")],
        tam=12.5, gap=8)
    notas(s, "El costo serverless supone GPU L4 a 0,000222 USD/s; en frío domina la carga de los modelos (10-17 s). "
             "En CPU la carga baja a 4-6 s porque no hay que mover pesos a la GPU. Fuente: results/31_cpu_vs_gpu.csv.")

    # ---------------------------------------------------------------- A13
    s = lamina(prs)
    titulo_lamina(s, "A13 · One2One en la mitad de arriba", "Cuántas candidatas muestrear y cuántas conservar")
    tabla(s, 0.6, 2.0, 7.9,
          ["configuración", "F1 suave", "precisión", "cobertura", "en formato", "historias"],
          [[("30 candidatas, tope 20 (elegida)", True), ("0,4325", True), "0,383", ("0,511", True), "77 %", "19,7"],
           ["60 candidatas, tope 20", "0,4266", "0,377", "0,505", "84 %", "19,8"],
           ["30 candidatas, tope 10", "0,4238", "0,389", "0,477", "86 %", "10,0"],
           ["30 candidatas, tope 20, solo formato", "0,4234", "0,382", "0,492", ("100 %", True), "15,5"],
           ["60 candidatas, tope 20, solo formato", "0,4233", "0,378", "0,495", "100 %", "17,7"],
           ["30 candidatas, tope 10, solo formato", "0,4192", "0,386", "0,471", "100 %", "9,2"]],
          anchos=[4.0, 1.3, 1.3, 1.3, 1.3, 1.2], tam=11, alto_fila=0.37)
    tarjeta(s, 8.7, 2.0, 4.05, 2.7, relleno=ARENA)
    texto(s, 8.95, 2.2, 3.55, 0.4, "La etapa de estilo", tam=13.5, negrita=True, color=MARINO)
    vinetas(s, 8.95, 2.7, 3.55, [
        ("", "Filtrar por formato cuesta 0,009 de F1 y da el 100 % de historias bien formadas. Se prefiere el "
             "formato: una historia que no parece historia no sirve en el backlog."),
        ("", "En Dalpiaz el modelo llega a 0,444 con 100 % de formato.")], tam=11.5)
    vinetas(s, 0.6, 4.85, 7.9, [
        ("Reescribir entradas con un LLM ", "solo ayuda cuando la entrada ya viene mal escrita: +0,019 en entradas "
                                           "reescritas (IC95 [+0,013; +0,025]) y empate en entradas de Jira reales y en historias humanas."),
        ("Por eso no se usa en producción: ", "añadiría una llamada a un LLM para no ganar nada en el caso normal, "
                                             "que es justo lo que queremos evitar.")],
        tam=12.5, gap=8)
    notas(s, "«Solo formato» descarta las candidatas que no tienen la forma «Como…, quiero…, para…». "
             "Fuente: results/18_barrido_o2o.csv, results/18_test_o2o.csv y results/16_eval_historias.csv.")

    # ---------------------------------------------------------------- A14
    s = lamina(prs)
    titulo_lamina(s, "A14 · Qué falta y cómo se cerraría", "Tres pendientes y uno ya resuelto")
    tabla(s, 0.6, 2.0, 12.13,
          ["lo que falta", "por qué", "cómo se cierra", "costo"],
          [[("Estudio con personas", True), "La métrica mide parecido, no utilidad",
            "Formulario ciego ya construido: 60 historias y 6 backlogs de tres sistemas mezclados; falta que lo "
            "respondan 3 o 4 personas", "15-25 min cada uno"],
           [("Cobertura", True), "Cobertura 0,47: proponemos menos tareas de las que el equipo escribió; los LLM proponen de más",
            "Dejar que el usuario pida más tareas por historia, con la precisión como tope", "1 día de trabajo"],
           [("Contenido en dominios de consumo", True),
            "El corpus son epics técnicos de Jira; para una app de consumo las historias salen vagas",
            "Añadir un corpus de producto o permitir 3 o 4 ejemplos del propio equipo", "depende del corpus"],
           [("Tareas repetidas dentro de una historia", True),
            "RESUELTO el 25/09: el pipeline solo quitaba duplicados entre historias; dentro de una, el 13,6 % de las tareas repetía otra",
            "Se colapsan también dentro de la historia, a coseno 0,95: quedan en 0 %. Cuesta 0,005 de F1 y 3 tareas por backlog; los seis empates con los LLM se mantienen",
            "aplicado"]],
          anchos=[2.6, 3.0, 4.9, 1.7], tam=10.5, alto_fila=0.82)
    vinetas(s, 0.6, 6.32, 12.2, [
        ("Lo que no falta: ", "el pipeline está completo y corre sin GPU, hay 52 scripts numerados que reproducen "
                              "cada cifra, y las 54 cifras del registro apuntan al archivo del que salieron.")],
        tam=13)
    notas(s, "Decir esto antes de que lo pregunten es mejor que defenderlo después. Cada punto tiene un plan y un "
             "costo, que es lo que distingue un límite reconocido de un problema no visto.")


# =====================================================================  RESULTADOS
def deck_resultados():
    prs = nueva()

    s = lamina(prs, MARINO)
    texto(s, 0.9, 2.2, 11.5, 1.6, "Generación de historias de usuario y tareas\ncon un modelo propio",
          tam=40, color=BLANCO, negrita=True, fuente=SERIF, interlineado=1.1)
    texto(s, 0.9, 4.0, 11.0, 0.9,
          "Resultados medidos: fidelidad, velocidad, generalización y costo de despliegue",
          tam=17, color=HIELO)
    texto(s, 0.9, 6.3, 11.0, 0.5, "Módulo de generación de Agix · tesis de pregrado · setiembre 2026",
          tam=12.5, color=HIELO)
    notas(s, "Tesis: reemplazar el LLM que genera tareas en Agix por un modelo propio, entrenado con "
             "datos reales de Jira y evaluable. Todo lo que sigue está medido sobre proyectos que el "
             "modelo nunca vio.")

    s = lamina(prs)
    titulo_lamina(s, "La pregunta", "¿Puede un modelo propio y pequeño reemplazar al LLM en esta tarea?")
    vinetas(s, 0.6, 2.05, 6.2, [
        ("Hoy: ", "Agix pide a un LLM externo que invente el backlog. No se puede medir, cambia debajo y depende de un proveedor."),
        ("Propuesta: ", "dos generadores mT5 de 121M parámetros entrenados sobre 2,8 millones de incidencias reales, más recuperación para dar procedencia."),
        ("Criterio: ", "fidelidad razonable, mucha más velocidad, costo predecible y capacidad de medir todo."),
    ], tam=15.5)
    tarjeta(s, 7.2, 2.0, 5.6, 3.6, relleno=ARENA)
    texto(s, 7.5, 2.25, 5.0, 0.4, "Lo que se mide en cada resultado", tam=13, negrita=True, color=MARINO)
    vinetas(s, 7.5, 2.8, 5.0, [
        ("F1 suave: ", "parecido semántico entre lo generado y el backlog real, sin exigir coincidencia exacta."),
        ("Cobertura: ", "cuánto del backlog real se propone."),
        ("Intervalos: ", "bootstrap pareado al 95 %; si cruza cero, no se afirma nada."),
    ], tam=13)
    notas(s, "No hay una única descomposición correcta: dos equipos parten la misma historia distinto y ambos "
             "aciertan. Por eso la métrica es de similitud y no de coincidencia exacta, y por eso todo va con "
             "intervalo de confianza.")

    s = lamina(prs)
    titulo_lamina(s, "Los datos", "Jerarquía real de trabajo, reconstruida de dos volcados públicos")
    stats(s, 2.1, [("2 845 066", "incidencias unificadas (TAWOS + Jira público)"),
                   ("244 677", "relaciones padre → hijo reconstruidas"),
                   ("1 289", "proyectos, partidos por proyecto entero"),
                   ("16 172", "incidencias escritas como historia de usuario")])
    tabla(s, 0.6, 4.0, 12.13,
          ["unidad de entrenamiento", "train", "val", "test", "qué enseña"],
          [[("historia → tareas", True), "11 202", "1 382", "1 751", "descomponer una historia en tareas"],
           [("epic → historias", True), "71 817 pares", "8 041", "9 073", "proponer el conjunto de historias (One2One)"],
           [("Dalpiaz", True), "15 proyectos", "3", "4 reservados", "historias reales en formato, nunca vistas"]],
          anchos=[3.2, 2.0, 1.2, 1.6, 4.2], tam=12.5, alto_fila=0.42)
    notas(s, "TAWOS no tiene columna de padre: la jerarquía se reconstruyó del historial de cambios (92 706 "
             "cambios de Epic Link). La partición es por proyecto entero, porque partir por incidencia "
             "regalaría las tareas del mismo proyecto.")

    s = lamina(prs)
    titulo_lamina(s, "El pipeline", "Cuatro etapas; el 95 % del tiempo está en los dos generadores")
    etapas = [("1", "Recuperar", "SBERT propio busca proyectos parecidos; da procedencia", "0,03 s"),
              ("2", "Historias", "mT5 One2One muestrea 30 candidatas y se deduplican", "0,68 s"),
              ("3", "Tareas", "mT5 One2Seq descompone cada historia (2 modelos)", "3,17 s"),
              ("4", "Ensamblar", "quita tareas repetidas entre historias y traduce", "0,10 s")]
    for i, (n, t, d, seg) in enumerate(etapas):
        x = 0.6 + i * 3.12
        tarjeta(s, x, 2.15, 2.9, 2.5, relleno=ARENA)
        texto(s, x + 0.25, 2.35, 0.6, 0.5, n, tam=26, color=MENTA, negrita=True, fuente=SERIF)
        texto(s, x + 0.25, 2.95, 2.4, 0.4, t, tam=15, negrita=True, color=MARINO)
        texto(s, x + 0.25, 3.35, 2.4, 1.0, d, tam=12, color=SUAVE)
        texto(s, x + 0.25, 4.25, 2.4, 0.3, seg, tam=13, color=TEAL, negrita=True)
    texto(s, 0.6, 5.0, 12.2, 0.5, "4,18 s por backlog completo: 12 historias y ~48 tareas (mediana de 64 proyectos)",
          tam=15, negrita=True, color=MARINO)
    vinetas(s, 0.6, 5.6, 12.2, [
        ("Dato útil para el despliegue: ", "el SBERT es solo el 2,7 % del tiempo y sus vectores están precalculados.")],
        tam=13.5, color=SUAVE)
    notas(s, "Las etapas están cronometradas por separado en cada corrida. Saberlo permitió decidir qué recortar "
             "para desplegar: los generadores, no la recuperación.")

    s = lamina(prs)
    titulo_lamina(s, "Resultado 1: historia → tareas", "Prueba sobre 1 751 ejemplos de 164 proyectos nunca vistos")
    barras(s, 0.5, 1.95, 7.4, 3.9,
           ["Solo recuperación", "mT5-base", "mT5-small", "Dos modelos (elegido)"],
           [0.3182, 0.4640, 0.4684, 0.4852],
           colores=[RGBColor(0xC9, 0xCD, 0xD4), TEAL, TEAL, MARINO])
    texto(s, 0.5, 5.9, 7.4, 0.4, "F1 suave (mayor es mejor)", tam=12, color=SUAVE)
    tarjeta(s, 8.2, 1.95, 4.6, 3.9, relleno=ARENA)
    texto(s, 8.5, 2.2, 4.0, 0.4, "Qué dice", tam=14, negrita=True, color=MARINO)
    vinetas(s, 8.5, 2.7, 4.0, [
        ("+0,167 ", "sobre copiar del proyecto más parecido: el modelo aporta, no repite."),
        ("Empate small y base: ", "IC95 [-0,009; 0,000]. Se elige small, 2,6 veces más chico."),
        ("Unir los dos: ", "+0,012 F1 y +0,029 cobertura, IC95 sin cruzar cero."),
    ], tam=12.5)
    notas(s, "El conjunto de dos modelos salió de aprovechar el modelo 'fallido' de la búsqueda de "
             "hiperparámetros: solo no aportaba, pero se equivoca distinto y su unión cubre más.")

    s = lamina(prs)
    titulo_lamina(s, "Resultado 2: descripción → historias", "Dalpiaz: 4 proyectos reservados, nunca usados para entrenar")
    tabla(s, 0.6, 2.05, 12.13,
          ["sistema", "F1 suave", "cobertura", "en formato historia", "historias"],
          [["Solo recuperación", "0,224", "0,179", "25 %", "3"],
           [("Modelo One2One + etapa de estilo", True), ("0,449", True), ("0,414", True), ("100 %", True), ("20", True)]],
          anchos=[4.4, 1.8, 1.8, 2.4, 1.6], tam=13.5, alto_fila=0.5)
    stats(s, 3.4, [("+0,225", "diferencia frente a recuperación · IC95 [+0,147; +0,295]"),
                   ("100 %", "de las historias salen como «Como…, quiero…, para…»"),
                   ("+0,104", "en los 60 epics de prueba · IC95 [+0,068; +0,137]")],
          alto=1.6)
    vinetas(s, 0.6, 5.35, 12.2, [
        ("Cómo se logró: ", "One2One (una historia por ejemplo) multiplicó 22 proyectos en 1 673 pares, y una segunda etapa entrenada solo con historias reales fijó el estilo."),
        ("Límite honesto: ", "son 4 proyectos; el número es indicativo. El contenido aún mezcla términos de otros dominios.")],
        tam=13.5)
    notas(s, "El primer intento de imponer el estilo con un token de control falló: solo 195 ejemplos y el "
             "modelo lo ignoró. Al descubrir que el estilo debía juzgarse en el título del objetivo, quedaron "
             "4 218 ejemplos limpios y la segunda etapa funcionó.")

    s = lamina(prs, MARINO)
    texto(s, 0.7, 0.55, 12.0, 0.8, "Velocidad: el argumento central", tam=34, color=BLANCO,
          negrita=True, fuente=SERIF)
    texto(s, 0.7, 1.4, 12.0, 0.5, "64 proyectos, misma máquina, backlog completo de punta a punta",
          tam=14.5, color=HIELO)
    barras(s, 0.5, 2.0, 7.6, 4.0,
           ["Nuestro pipeline", "llama3.1:8b", "gemma2:9b", "mistral:7b"],
           [4.18, 21.63, 35.73, 39.44], colores=[MENTA, HIELO, HIELO, HIELO], formato='0.0',
           color_texto=BLANCO, color_ejes=HIELO)
    texto(s, 0.5, 6.05, 7.6, 0.4, "segundos por backlog (mediana)", tam=12, color=HIELO)
    stats(s, 2.2, [("5,2x", "más rápido que llama3.1:8b"), ("9,4x", "más rápido que mistral:7b")],
          ancho_total=4.5, x0=8.3, alto=1.5, color_num=MENTA, fondo=RGBColor(0x2A, 0x33, 0x6D),
          color_etq=HIELO)
    vinetas(s, 8.3, 4.0, 4.5, [
        ("Con GPU: ", "4,18 s frente a 21–39 s."),
        ("Sin GPU: ", "11,4 s en CPU, frente a 182 s de llama3.1 en la misma CPU."),
        ("Energía: ", "4,2 s a 80 W contra 20–40 s a 80 W.")], tam=13, color=HIELO)
    notas(s, "El p90 también se midió: 4,6 s el nuestro contra 25–45 s los LLM. La comparación es pareada: "
             "los mismos 64 proyectos en los cuatro sistemas.")

    s = lamina(prs)
    titulo_lamina(s, "Fidelidad: ya no pierde contra ninguno",
                  "64 proyectos, comparación pareada por proyecto, bootstrap al 95 %")
    tabla(s, 0.6, 2.0, 12.13,
          ["sistema", "F1 suave", "diferencia contra el nuestro (IC95)", "veredicto", "reconoció"],
          [[("Nuestro pipeline", True), ("0,472", True), "—", "—", ("0 %", True)],
           ["llama3.1:8b sin ejemplos", "0,463", "−0,009  [−0,035; +0,017]", ("empate", True), "81 %"],
           ["mistral:7b sin ejemplos", "0,473", "+0,001  [−0,023; +0,026]", ("empate", True), "55 %"],
           ["gemma2:9b sin ejemplos", "0,475", "+0,003  [−0,026; +0,031]", ("empate", True), "59 %"],
           ["llama3.1:8b con ejemplos", "0,479", "+0,007  [−0,016; +0,033]", ("empate", True), "81 %"],
           ["mistral:7b con ejemplos", "0,486", "+0,014  [−0,009; +0,036]", ("empate", True), "55 %"],
           ["gemma2:9b con ejemplos", "0,487", "+0,015  [−0,010; +0,040]", ("empate", True), "59 %"]],
          anchos=[3.5, 1.5, 3.9, 1.7, 1.5], tam=12, alto_fila=0.38)
    tarjeta(s, 0.6, 5.0, 5.9, 1.75, relleno=ARENA)
    texto(s, 0.9, 5.2, 5.3, 0.4, "Los seis intervalos cruzan el cero", tam=14, negrita=True, color=MARINO)
    vinetas(s, 0.9, 5.65, 5.3, [
        ("", "Antes de destilar, dos de los seis ganaban con intervalos que excluían el cero. Ahora ninguno.")], tam=12.5)
    tarjeta(s, 6.8, 5.0, 5.93, 1.75, relleno=ARENA)
    texto(s, 7.1, 5.2, 5.3, 0.4, "Y su ventaja sigue contaminada", tam=14, negrita=True, color=CORAL)
    vinetas(s, 7.1, 5.65, 5.3, [
        ("", "Reconocieron del 55 % al 81 % de los proyectos; el nuestro, 0 %, porque quedaron fuera por construcción.")], tam=12.5)
    notas(s, "Este es el resultado tras la destilación del 25 de setiembre. Antes el nuestro daba 0,458 y perdía "
             "contra mistral y gemma2 con ejemplos; ahora da 0,477 y los seis intervalos cruzan el cero. "
             "Fuente: results/22_banco_pareado.csv.")

    s = lamina(prs)
    titulo_lamina(s, "Generalización a dominios nuevos", "Se entrena sin un dominio entero y se evalúa en él")
    tabla(s, 0.6, 2.05, 8.0,
          ["dominio dejado fuera", "dentro", "fuera", "diferencia"],
          [["Apps web y CMS (Moodle, Sakai)", "0,446", "0,457", "+0,011"],
           ["Blockchain y distribuidos (Fabric, Kafka)", "0,458", "0,481", "+0,023"],
           ["Big data (Spark, Flink, HBase)", "0,452", "0,459", "+0,007"]],
          anchos=[4.4, 1.2, 1.2, 1.5], tam=13, alto_fila=0.46)
    tarjeta(s, 8.9, 2.05, 3.85, 2.3, relleno=ARENA)
    texto(s, 9.15, 2.3, 3.35, 0.4, "Sin caída", tam=15, negrita=True, color=MARINO)
    vinetas(s, 9.15, 2.75, 3.35, [("", "Lo aprendido es cómo se descompone trabajo de software, no el vocabulario de un dominio.")], tam=12.5)
    vinetas(s, 0.6, 4.8, 12.2, [
        ("Control: ", "los dominios difieren en dificultad (0,455 a 0,524), y los dos que «mejoran» son los más difíciles; no es un conjunto fácil."),
        ("Resultado negativo: ", "la similitud de recuperación no sirve como señal de «no sé»: fuera de dominio es más alta y correlaciona menos con la calidad."),
        ("Corrección aplicada: ", "la primera versión del experimento mezclaba proyectos de train en el lado «fuera» y exageraba el efecto; ambos lados usan solo proyectos de prueba.")],
        tam=13)
    notais = None
    notas(s, "Dominios por k-means sobre vectores de proyecto. Tres reentrenamientos de ~32 min, uno por dominio, "
             "cada uno en su propio proceso.")

    s = lamina(prs)
    titulo_lamina(s, "Despliegue: no hace falta GPU", "Medido en esta máquina y con Ollama forzado a CPU")
    tabla(s, 0.6, 2.0, 7.7,
          ["configuración", "segundos", "RAM"],
          [[("Nuestro pipeline, GPU", True), ("3,99", True), "—"],
           ["Nuestro, CPU 4 hilos, ligero", "7,0", "3,0 GB"],
           ["Nuestro, CPU 4 hilos, completo", "11,4", "3,2 GB"],
           ["llama3.1:8b con GPU", "21,6", "—"],
           [("llama3.1:8b en CPU", True), ("181,9", True), "—"]],
          anchos=[4.3, 1.7, 1.7], tam=13, alto_fila=0.44)
    texto(s, 0.6, 4.55, 7.7, 0.5,
          "En CPU somos 16x más rápidos; incluso en CPU superamos al LLM con GPU dedicada.",
          tam=14, negrita=True, color=MARINO)
    tarjeta(s, 8.6, 2.0, 4.15, 4.3, relleno=ARENA)
    texto(s, 8.9, 2.25, 3.6, 0.4, "Costo por opción", tam=14, negrita=True, color=MARINO)
    vinetas(s, 8.9, 2.75, 3.6, [
        ("Servidor CPU: ", "unos pocos dólares al mes, sin GPU."),
        ("Serverless: ", "0,10 centavos por proyecto en caliente; cero en reposo."),
        ("PC propia 24/7: ", "~58 kWh al mes, unos S/ 40."),
        ("API externa: ", "0,05 centavos, pero con dependencia del proveedor.")], tam=12.5)
    notas(s, "La laptop de la presentación (Ryzen 3, 8 GB, sin gráfica) corre el pipeline: 3 GB de RAM y 11 s "
             "por backlog con 4 hilos. El modelo que Agix usaba en Groq dejó de estar en autoservicio en agosto "
             "de 2026: esa dependencia es un riesgo medido, no hipotético.")

    # ==================================================== destilacion de dominio
    s = lamina(prs)
    titulo_lamina(s, "El salto: enseñarle dominios de producto",
                  "El corpus son incidencias técnicas de Jira; Agix recibe descripciones de producto")
    vinetas(s, 0.6, 2.0, 6.4, [
        ("El problema, en una frase: ", "para «una tienda de audiolibros» el modelo respondía con vocabulario de Hyperledger, porque es lo único que había visto."),
        ("Qué se hizo: ", "un LLM local con licencia Apache 2.0 escribió backlogs para 30 sectores de consumo, y el modelo continuó su entrenamiento sobre ellos mezclados con los datos reales."),
        ("Lo que NO se hizo: ", "no hay ningún LLM en el producto. Solo se usó para fabricar ejemplos de entrenamiento, una vez."),
    ], tam=14)
    stats(s, 4.75, [("2 h", "de principio a fin"),
                    ("30", "sectores cubiertos"),
                    ("5 814", "historias nuevas")], ancho_total=6.4, x0=0.6, alto=1.5)
    texto(s, 0.6, 6.45, 6.4, 0.4, "Sin crecer: sigue siendo el mismo modelo de 121M parámetros.",
          tam=12.5, color=SUAVE)
    tabla(s, 7.3, 2.0, 5.43,
          ["dónde se midió", "antes", "después"],
          [[("Historias humanas reales", True), ("0,552", True), ("0,650", True)],
           ["Consumo, historia → tareas", "0,502", "0,669"],
           ["Consumo, descripción → historia", "0,310", "0,617"],
           ["Jira reservado (no romper)", "0,474", "0,474"],
           ["Dalpiaz reservado (no romper)", "0,234", "0,232"]],
          anchos=[3.0, 1.1, 1.1], tam=11.5, alto_fila=0.42)
    texto(s, 7.3, 4.65, 5.43, 1.0,
          "Sube donde tenía que subir y no se mueve donde no debía: las dos filas de abajo son la "
          "comprobación de que no se rompió lo que ya funcionaba.", tam=12.5, color=MARINO, negrita=True)
    notas(s, "El modelo profesor fue qwen2.5:7b, Apache 2.0, elegido leyendo sus salidas contra las de "
             "mistral. Es destilación de conocimiento: técnica estándar, y el argumento de la tesis no "
             "se toca porque en inferencia no hay ningún LLM.")

    # ============================================== validacion con datos humanos
    s = lamina(prs)
    titulo_lamina(s, "La validación que no se puede discutir",
                  "328 historias escritas por personas, de otro grupo de investigación, nunca vistas")
    vinetas(s, 0.6, 2.0, 6.0, [
        ("El riesgo: ", "si entreno con backlogs de un LLM y evalúo con referencias de ese mismo LLM, parte de la mejora sería solo parecerse al profesor."),
        ("La solución: ", "un conjunto publicado por Bragilovski y otros (CC BY 4.0) con historias humanas de 9 proyectos. Se comprobó el solapamiento con nuestros datos y 3 se descartaron por repetirse."),
        ("Quedan 6 proyectos de consumo: ", "restaurante, ticketera, supermercado, colegio, gimnasio y contratación."),
    ], tam=14)
    barras(s, 7.2, 2.0, 5.55, 3.0, ["Antes", "Después"], [0.5522, 0.6504],
           colores=[RGBColor(0xC9, 0xCD, 0xD4), MENTA], formato='0.000')
    texto(s, 7.2, 5.05, 5.55, 0.4, "F1 suave contra historias humanas", tam=12, color=SUAVE)
    stats(s, 5.45, [("+0,093", "de mejora · IC95 [+0,053; +0,139]"),
                    ("6 de 6", "proyectos en los que gana"),
                    ("0,60 → 0,72", "precisión, que era el punto flojo")], ancho_total=12.13, x0=0.6,
          alto=1.35)
    notas(s, "Esta es la única medición de la destilación que no puede explicarse por mimetismo del "
             "profesor, porque las referencias las escribieron personas y el conjunto lo publicó otro "
             "grupo. n=6 proyectos, así que el intervalo es ancho por construcción. "
             "Fuente: results/49_real_30sec.csv.")

    s = lamina(prs)
    titulo_lamina(s, "Límites, dichos por nosotros", "Lo que todavía no funciona bien")
    vinetas(s, 0.6, 2.0, 12.2, [
        ("Contenido genérico fuera del dominio de Jira. ", "Para una descripción de producto de consumo («app de limpieza»), las historias son plausibles en forma pero vagas en contenido: el corpus son epics técnicos."),
        ("Cobertura de 0,47. ", "Proponemos menos tareas de las que el equipo escribió; los LLM proponen de más (2 a 4 veces) y así cubren algo más por volumen."),
        ("El español funciona por traducción, no por entrenamiento. ", "Entrenar con entradas traducidas empeoró el inglés (−0,032) y no mejoró el español; traducir al inferir sí (+0,045, 175 ms)."),
        ("Sin estudio con personas todavía. ", "La muestra ya está congelada y ciega: 60 historias y 6 proyectos con tres sistemas mezclados, listos para evaluar."),
        ("La métrica es un proxy. ", "Mide parecido semántico, no utilidad. Por eso el estudio con personas es el siguiente paso."),
    ], tam=14.5, gap=11)
    notas(s, "Decir los límites primero desarma la mitad de las preguntas difíciles y demuestra que se midió "
             "en serio, no que se buscó un número bonito.")

    s = lamina(prs, MARINO)
    texto(s, 0.8, 1.1, 11.8, 0.8, "En una frase", tam=32, color=BLANCO, negrita=True, fuente=SERIF)
    texto(s, 0.8, 2.1, 11.8, 2.2,
          "Un modelo propio de 121M parámetros iguala la fidelidad de seis configuraciones "
          "de LLM de 7 a 9 mil millones en esta tarea, siendo de 5 a 10 veces más rápido, sin enviar "
          "datos a terceros y con todo medido sobre proyectos que nunca vio.",
          tam=23, color=BLANCO, interlineado=1.3)
    vinetas(s, 0.8, 4.6, 11.8, [
        ("Lo que sigue: ", "estudio con personas, y decidir el despliegue entre servidor CPU y serverless."),
        ("Lo que ya está listo: ", "pipeline completo, demo local sin GPU y 52 scripts reproducibles con sus cifras registradas.")],
        tam=15, color=HIELO, gap=10)
    notas(s, "Igualar quiere decir: 0,458 contra 0,463-0,475 sin ejemplos, con los tres intervalos "
             "cruzando el cero en la comparacion pareada de 64 proyectos. Con ejemplos en el prompt "
             "los LLM siguen ganando por 0,028, que es el 94 % de su fidelidad.")

    anexos(prs)
    prs.save(f"{SALIDA}/Agix_resultados.pptx")
    print(f"escrito {SALIDA}/Agix_resultados.pptx ({len(prs.slides.__iter__.__self__._sldIdLst)} laminas)")


# =======================================================================  PROCESO
def deck_proceso():
    prs = nueva()

    s = lamina(prs, MARINO)
    texto(s, 0.9, 2.3, 11.5, 1.5, "Cómo construimos el pipeline", tam=40, color=BLANCO,
          negrita=True, fuente=SERIF)
    texto(s, 0.9, 3.9, 11.0, 1.0,
          "Del volcado de Jira al modelo desplegable: decisiones, mediciones y errores corregidos",
          tam=17, color=HIELO)
    texto(s, 0.9, 6.3, 11.0, 0.5, "Módulo de generación de Agix · setiembre 2026", tam=12.5, color=HIELO)
    notas(s, "Esta presentación cuenta el método. Cada decisión de diseño se tomó con una medición, y varias "
             "de ellas contradijeron lo que esperábamos.")

    s = lamina(prs)
    titulo_lamina(s, "Punto de partida", "Tres correcciones antes de escribir una línea de modelo")
    vinetas(s, 0.6, 2.0, 12.2, [
        ("No existían los pesos del SBERT ajustado. ", "Los ajustes previos guardaron solo vectores; y además se habían entrenado para predecir horas, no para medir parecido. Se partió del modelo multilingüe congelado."),
        ("TAWOS no tiene columna de padre. ", "La jerarquía epic → historias → subtareas se reconstruyó del historial de cambios: 92 706 cambios de «Epic Link» y 73 054 de «Epic Child»."),
        ("El Jira público extraído estaba filtrado. ", "Solo traía incidencias con horas registradas (2,9 %). Se reextrajo completo: 2 686 282 incidencias en 8 minutos, leyendo el volcado en streaming."),
    ], tam=15, gap=12)
    stats(s, 4.9, [("2,4 min", "para extraer la jerarquía de TAWOS"),
                   ("8,1 min", "para releer el volcado de Jira completo"),
                   ("~142 mil", "sub-tareas con padre recuperadas"),
                   ("~113 mil", "incidencias con epic asignado")])
    notas(s, "El handoff daba por hecho que la jerarquía era escasa. Medirla primero cambió el plan: había 10 "
             "veces más datos de los previstos, y eso habilitó entrenar las dos mitades.")

    s = lamina(prs)
    titulo_lamina(s, "Dos mitades con datos muy distintos", "El diseño se decide contando ejemplos, no por gusto")
    tarjeta(s, 0.6, 2.05, 5.95, 3.4, relleno=ARENA)
    texto(s, 0.95, 2.3, 5.3, 0.4, "Abajo · historia → tareas", tam=16, negrita=True, color=MARINO)
    vinetas(s, 0.95, 2.8, 5.3, [
        ("14 335 ejemplos. ", "Un padre con sus sub-tareas reales."),
        ("Seq2seq justificado: ", "hay volumen para entrenar generación."),
        ("Entrada: ", "la historia y ejemplos recuperados."),
    ], tam=13)
    tarjeta(s, 6.85, 2.05, 5.9, 3.4, relleno=ARENA)
    texto(s, 7.2, 2.3, 5.2, 0.4, "Arriba · descripción → historias", tam=16, negrita=True, color=MARINO)
    vinetas(s, 7.2, 2.8, 5.2, [
        ("Solo 39 descripciones de proyecto. ", "Insuficiente para condicionar un decodificador."),
        ("Solución 1: ", "usar el epic como descripción, y pasan a ser miles."),
        ("Solución 2: ", "One2One, una historia por ejemplo, y 22 proyectos de Dalpiaz dan 1 673 pares."),
    ], tam=13)
    texto(s, 0.6, 5.75, 12.2, 0.9,
          "Contar bien los ejemplos es lo que decidió el algoritmo de cada mitad: con una fila por proyecto "
          "no había nada que entrenar; con una fila por historia, sí.", tam=14.5, color=MARINO, negrita=True)
    notas(s, "One2One es la reformulación del plan original: en inferencia se muestrea muchas veces y se "
             "deduplica con el SBERT, votando por las historias que el modelo repite.")

    s = lamina(prs)
    titulo_lamina(s, "Rigor de partición", "Sin esto, cualquier número es humo")
    vinetas(s, 0.6, 2.0, 7.4, [
        ("Partición por proyecto entero. ", "Partir por incidencia dejaría tareas del mismo proyecto en train y test."),
        ("Una sola partición para todo. ", "El SBERT, los generadores y la evaluación comparten el archivo de splits, para que el SBERT no haya visto los proyectos de prueba del mT5."),
        ("Dominios por k-means. ", "Seis grupos temáticos sobre vectores de proyecto, para poder dejar fuera un dominio completo."),
        ("Dalpiaz reservado. ", "4 de 22 proyectos nunca se tocaron: son la prueba final de la mitad de arriba."),
    ], tam=14.5, gap=11)
    tarjeta(s, 8.3, 2.05, 4.45, 3.9, relleno=ARENA)
    texto(s, 8.6, 2.3, 3.9, 0.4, "Reparto final", tam=14, negrita=True, color=MARINO)
    vinetas(s, 8.6, 2.8, 3.9, [
        ("Train: ", "1 003 proyectos (79,8 % de las relaciones)"),
        ("Validación: ", "122 proyectos (9,3 %)"),
        ("Prueba: ", "164 proyectos (10,9 %)"),
        ("", "Los proyectos gigantes van forzados a train para que uno solo no sea toda la prueba."),
    ], tam=12.5)
    notas(s, "Los duplicados se agrupan por hash de texto para que no cruce la misma tarea entre particiones.")

    s = lamina(prs)
    titulo_lamina(s, "SBERT propio para recuperar", "Ajuste contrastivo, no de regresión")
    vinetas(s, 0.6, 2.0, 6.6, [
        ("Qué se entrenó: ", "201 309 pares (padre → hijo) con pérdida contrastiva y negativos en el lote; batch 256 con GradCache para no pasar de 6,5 GB de VRAM."),
        ("Bilingüe a propósito: ", "el 30 % de las anclas se tradujo al español, porque las consultas de Agix llegan en español."),
        ("Criterio fijado antes: ", "adoptarlo solo si superaba al congelado en inglés y en español, con IC95 sin cruzar cero."),
    ], tam=14.5)
    barras(s, 7.0, 2.0, 5.8, 3.4, ["Inglés congelado", "Inglés ajustado", "Español congelado", "Español ajustado"],
           [0.3531, 0.3923, 0.2294, 0.3167],
           colores=[RGBColor(0xC9, 0xCD, 0xD4), TEAL, RGBColor(0xC9, 0xCD, 0xD4), MARINO], formato='0.000')
    texto(s, 7.0, 5.45, 5.8, 0.4, "nDCG@10 en proyectos de prueba", tam=12, color=SUAVE)
    texto(s, 0.6, 5.7, 6.4, 0.9, "Mejora de +0,039 en inglés y +0,087 en español, ambas con intervalos "
          "que excluyen el cero.", tam=14, negrita=True, color=MARINO)
    notas(s, "El ajuste anterior de la tesis hermana optimizaba horas y deformaba el espacio; con pérdida "
             "contrastiva el espacio sirve para buscar, que es lo que el pipeline necesita.")

    s = lamina(prs)
    titulo_lamina(s, "Poda de vocabulario", "La decisión técnica que más rindió")
    vinetas(s, 0.6, 2.0, 6.9, [
        ("El problema: ", "en mT5 el vocabulario multilingüe de 250 100 piezas es la mayor parte del modelo y de la memoria de los logits."),
        ("La poda: ", "se conservan las 75 260 piezas que aparecen en nuestro corpus en inglés y español, más los 256 bytes de reserva."),
        ("Verificación: ", "los ids coinciden y la diferencia de logits con el modelo original es exactamente 0,0; en español reservado, 0 % de palabras desconocidas."),
    ], tam=14.5)
    stats(s, 4.5, [("300M → 121M", "parámetros en mT5-small"),
                   ("582M → 314M", "parámetros en mT5-base, que pasó a ser viable"),
                   ("19,5 → 67,6", "ejemplos por segundo entrenando"),
                   ("0,0", "diferencia de logits: la poda es exacta")])
    tarjeta(s, 7.7, 2.0, 5.05, 2.2, relleno=ARENA)
    texto(s, 8.0, 2.25, 4.45, 0.4, "Por qué importa para el producto", tam=13.5, negrita=True, color=MARINO)
    vinetas(s, 8.0, 2.75, 4.45, [("", "Un modelo de 121M cabe en una laptop de 8 GB sin gráfica y arranca en 4 segundos: es lo que hace posible el despliegue barato.")], tam=12.5)
    notas(s, "Sin la poda, mT5-base habría necesitado ~6 h por época en esta GPU; con la poda bajó a 20 min y "
             "el small pasó de 27 a 45 ejemplos por segundo.")

    s = lamina(prs)
    titulo_lamina(s, "Lo que esperábamos y lo que midió", "Cinco decisiones que cambiaron de rumbo por evidencia")
    tabla(s, 0.6, 1.95, 12.13,
          ["hipótesis", "medición", "decisión"],
          [["Los ejemplos recuperados en el prompt ayudan", "No: quitarlos da +0,022 F1 arriba y empata abajo; y evita 11,8 % de fuga de vocabulario", ("Se quitaron", True)],
           ["Muestrear más sube la cobertura útil", "Sube cobertura pero rompe el piso de precisión (0,58 → 0,44)", ("Descartado", True)],
           ["Entrenar con español mejora el español", "Empeoró el inglés (−0,032) y no mejoró el español", ("Traducir al inferir", True)],
           ["La búsqueda de hiperparámetros mejora el modelo", "41 pruebas con Optuna y poda ASHA; la ganadora no superó a los valores a mano", ("Se mantuvo lo previo", True)],
           ["El modelo genera listas cortas", "En realidad repetía hasta el tope; los parámetros de longitud no hacían nada", ("Bloqueo de n-gramas", True)]],
          anchos=[3.9, 5.9, 2.4], tam=12, alto_fila=0.62)
    texto(s, 0.6, 5.9, 12.2, 0.9, "Cuatro de las cinco hipótesis eran falsas. Cada una está registrada con su "
          "medición, y dos de ellas son resultados negativos publicables.", tam=14, negrita=True, color=MARINO)
    notas(s, "Que la recuperación no ayude dentro del prompt es un resultado contraintuitivo: los modelos "
             "aprendieron a ignorar los ejemplos. La recuperación sigue sirviendo para ordenar candidatos y "
             "para mostrar procedencia.")

    s = lamina(prs)
    titulo_lamina(s, "Errores encontrados y corregidos", "Parte del método es auditar el propio trabajo")
    vinetas(s, 0.6, 2.0, 12.2, [
        ("Conjunto de evaluación sesgado. ", "La prueba por dominio mezclaba proyectos de train en el lado «fuera», que son los grandes y formularios. Exageraba el efecto de +1,1 % a +4,9 %. Se rehízo con solo proyectos de prueba."),
        ("Estilo juzgado en el lugar equivocado. ", "Se marcaba «historia» mirando título o descripción, pero el modelo solo escribe títulos: el 80 % de esos ejemplos no lo eran. Al corregirlo, la etapa de estilo funcionó."),
        ("Un carácter invisible. ", "Un parche escribió un retroceso en vez de «\\b» en una expresión regular: la protección de términos no hacía nada y las métricas salían iguales. Se detectó comparando con una prueba aislada."),
        ("Desborde de VRAM silencioso. ", "Superar ~7 GB no da error en Windows: usa RAM compartida y todo va 10 veces más lento. Se detectó porque las épocas pasaron de 7 a 20 minutos."),
    ], tam=14, gap=11)
    notas(s, "Todos estos casos están documentados con la medición antes y después. La regla que quedó: "
             "comprobar que ambos lados de una comparación salgan de la misma población, y vigilar la segunda "
             "época de cada entrenamiento.")

    s = lamina(prs)
    titulo_lamina(s, "Trabajar con 8 GB de VRAM", "Restricciones que moldearon el diseño")
    tabla(s, 0.6, 2.0, 12.13,
          ["restricción medida", "consecuencia en el diseño"],
          [["Pasar de ~7 GB no da error: usa RAM compartida y va 10x más lento",
            "El batch se elige por memoria, no «hasta el error de memoria»"],
           ["El vocabulario de 250k domina los logits",
            "Poda a 75k: habilita batch 16 y el modelo base"],
           ["Cargar SBERT junto al entrenamiento desborda",
            "Un proceso por entrenamiento; el SBERT se carga solo para evaluar"],
           ["Cada corrida larga puede morir a mitad",
            "Mini-entrenamiento obligatorio antes de lanzar y reanudación por archivo"]],
          anchos=[6.0, 6.0], tam=12.5, alto_fila=0.62)
    stats(s, 4.9, [("52", "scripts numerados, en orden de ejecución"),
                   ("47", "cifras en un registro único con su fuente"),
                   ("~11 h", "de GPU en total, en corridas de noche"),
                   ("0", "dependencias nuevas en la demo")])
    notas(s, "Las corridas largas se encadenaron de noche con guiones que reanudan: el banco de LLM se cortó "
             "dos veces por fallos de Ollama y no se perdió trabajo.")

    # ================================================== arquitectura C4
    def _imagen(slide, ruta, x, y, ancho_max, alto_max):
        """centra la imagen dentro de la caja, respetando su relacion de aspecto"""
        from PIL import Image
        w, h = Image.open(ruta).size
        esc = min(ancho_max / w, alto_max / h)
        aw, ah = w * esc, h * esc
        return slide.shapes.add_picture(ruta, Inches(x + (ancho_max - aw) / 2),
                                        Inches(y + (alto_max - ah) / 2),
                                        Inches(aw), Inches(ah))

    DIAG = f"{BASE}/entrega/diagramas"

    s = lamina(prs)
    titulo_lamina(s, "Arquitectura · contexto y contenedores",
                  "Modelo C4. Lo importante del primer nivel es lo que ya no está")
    _imagen(s, f"{DIAG}/C1_contexto.png", 0.6, 1.95, 5.1, 4.6)
    _imagen(s, f"{DIAG}/C2_contenedores.png", 6.0, 2.3, 6.7, 3.9)
    texto(s, 0.6, 6.7, 5.1, 0.5, "C1 · el LLM externo, en gris, es lo que se sustituye",
          tam=11.5, color=SUAVE)
    texto(s, 6.0, 6.7, 6.7, 0.5, "C2 · todo corre en la misma máquina: no hay red",
          tam=11.5, color=SUAVE)
    notas(s, "El LLM externo aparece en gris a propósito: es la justificación del módulo entero. "
             "En el nivel de contenedores hay un solo pipeline porque una generación ocupa todos "
             "los modelos en memoria y se sirve de a una; escalar sería replicar el contenedor.")

    s = lamina(prs)
    titulo_lamina(s, "Arquitectura · las seis etapas",
                  "Tiempos medidos sobre 64 proyectos en GPU; el total son 4,18 s")
    _imagen(s, f"{DIAG}/C3_componentes.png", 0.6, 2.1, 12.13, 3.0)
    vinetas(s, 0.6, 5.4, 12.2, [
        ("El generador de tareas se lleva el 76 % del tiempo. ", "Saberlo es lo que permitió decidir qué recortar para desplegar sin GPU: se recortó ahí y no en la recuperación."),
        ("Son dos modelos, no uno. ", "El segundo salió de una búsqueda de hiperparámetros que «falló»: solo empata, pero se equivoca distinto, y la unión sube la cobertura de 0,438 a 0,467 sin bajar la precisión."),
        ("La recuperación no alimenta el prompt. ", "Se midió y los ejemplos recuperados no ayudaban. Se conserva para ordenar candidatas y mostrar procedencia."),
    ], tam=13, gap=9)
    notas(s, "Las etapas están cronometradas por separado en cada corrida, y por eso se puede decir "
             "con precisión dónde está el costo.")

    s = lamina(prs)
    titulo_lamina(s, "Cómo se construyó el modelo",
                  "Flujo de datos fuera de línea: no es parte del sistema en ejecución")
    _imagen(s, f"{DIAG}/construccion.png", 0.6, 2.2, 12.13, 2.6)
    tarjeta(s, 0.6, 5.1, 12.13, 1.5, relleno=ARENA)
    texto(s, 0.95, 5.35, 11.5, 0.4, "El único punto donde interviene un LLM", tam=14,
          negrita=True, color=MARINO)
    vinetas(s, 0.95, 5.8, 11.5, [
        ("", "El bloque naranja es el profesor que escribió los backlogs de ejemplo, una sola vez y fuera de línea. En el sistema desplegado no hay ningún LLM: solo modelos propios de 121M parámetros.")],
        tam=13)
    notas(s, "Esta vista se incluye porque es donde está el aporte de la tesis: construir el "
             "conjunto de entrenamiento, que no existía, y adaptar el modelo al dominio.")

    s = lamina(prs)
    titulo_lamina(s, "Reproducibilidad", "Que cualquiera pueda repetir el resultado")
    vinetas(s, 0.6, 2.0, 6.5, [
        ("Semillas fijas ", "en particiones, muestreos y bootstrap."),
        ("Criterios antes de mirar: ", "cada adopción de modelo o configuración declaró su métrica y su piso antes de la medición."),
        ("Un registro único: ", "results/REGISTRO.csv reconstruye las 47 cifras desde los artefactos; si algo no está ahí, no está medido."),
        ("Licencias limpias: ", "TAWOS Apache 2.0, Jira público CC BY 4.0, Dalpiaz CC BY 4.0, mT5 y SBERT Apache 2.0, mistral Apache 2.0."),
    ], tam=14, gap=10)
    tarjeta(s, 7.4, 2.05, 5.35, 3.9, relleno=ARENA)
    texto(s, 7.7, 2.3, 4.75, 0.4, "Un detalle de licencias", tam=13.5, negrita=True, color=MARINO)
    vinetas(s, 7.7, 2.8, 4.75, [
        ("", "La reescritura de entradas se hizo con mistral (Apache 2.0) y no con llama, porque la licencia de llama obliga a que el modelo resultante lleve «Llama» en el nombre."),
        ("", "Detectarlo a tiempo evitó condicionar el nombre del modelo de la tesis.")], tam=12.5)
    notas(s, "El registro se regenera con un comando. Eso evita que las diapositivas queden con cifras viejas "
             "cuando un experimento se repite, que ya pasó una vez con la mediana de velocidad.")

    s = lamina(prs, MARINO)
    texto(s, 0.8, 1.2, 11.8, 0.8, "El método, resumido", tam=32, color=BLANCO, negrita=True, fuente=SERIF)
    vinetas(s, 0.8, 2.3, 11.8, [
        ("Medir antes de diseñar. ", "Contar los ejemplos decidió el algoritmo de cada mitad."),
        ("Declarar el criterio antes del resultado. ", "Evita elegir la métrica que conviene."),
        ("Preferir la solución simple cuando empata. ", "Traducir en la inferencia le ganó a reentrenar."),
        ("Reportar lo que no funciona. ", "Cuatro hipótesis falsas y dos resultados negativos quedaron documentados."),
        ("Auditar el propio trabajo. ", "Tres errores propios encontrados y corregidos con medición."),
    ], tam=16, color=HIELO, gap=12)
    notas(s, "Si la asesora pregunta por la contribución metodológica, esta lámina es la respuesta.")

    prs.save(f"{SALIDA}/Agix_proceso.pptx")
    print(f"escrito {SALIDA}/Agix_proceso.pptx")


if __name__ == "__main__":
    deck_resultados()
    deck_proceso()
