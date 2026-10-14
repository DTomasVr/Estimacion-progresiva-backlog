# -*- coding: utf-8 -*-
"""Historias de usuario del componente de ESTIMACION, corregidas contra results/.

Solo las HU de la tuberia de estimacion; las de generacion de historias y tareas
(HU-56 a HU-63, HU-68, HU-69, HU-70) se omiten a proposito.

Cada cifra que aparece en un criterio esta verificada contra un CSV de results/.
Las que no se pudieron verificar quedan senaladas en la hoja Revision.
"""
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

FUENTE = "Calibri"
VERDE = PatternFill("solid", fgColor="E2EFDA")
GRIS = PatternFill("solid", fgColor="D9D9D9")
AMAR = PatternFill("solid", fgColor="FFF2CC")
ROJO = PatternFill("solid", fgColor="FCE4D6")
B = Border(*[Side(style="thin", color="BFBFBF")] * 4)
DESTINO = "entrega/Historias-de-Usuario-Estimacion-corregidas.xlsx"

# (ID, rol, caracteristica, razon, [(n, titulo, dado, cuando, entonces), ...])
HUS = [
    ("HU-34", "planificador",
     "que el sistema me proponga cuanto durara una tarea a partir del historial de ESTE proyecto",
     "no estimar a ojo",
     [(1, "Sugerencia con historial suficiente",
       "Dado que el proyecto tiene tareas terminadas con tiempo real.",
       "Cuando creo una tarea nueva.",
       "Entonces el sistema propone un esfuerzo estimado con su banda y el numero de tareas cerradas que lo sostienen."),
      (2, "Historial de otros proyectos del mismo equipo",
       "Dado que el equipo tiene proyectos anteriores cerrados en la plataforma.",
       "Cuando se estima un proyecto nuevo.",
       "Entonces esos proyectos NO alimentan la sugerencia, porque se midio que el desplazamiento es del proyecto y no del equipo.")]),

    ("HU-35", "planificador",
     "ver en que se apoya la sugerencia del sistema",
     "poder confiar en ella o descartarla",
     [(1, "Explicabilidad del valor por tarea",
       "Dado que el sistema propone un valor probable para una tarea.",
       "Cuando consulto su detalle.",
       "Entonces veo las tareas mas parecidas del corpus que sostienen ese valor y su dispersion."),
      (2, "Explicabilidad del ajuste del proyecto",
       "Dado que el total del proyecto esta anclado en tareas cerradas.",
       "Cuando consulto el detalle del total.",
       "Entonces veo que el ajuste es un unico factor del proyecto calculado con TODAS las tareas cerradas, no con las mas parecidas a cada tarea.")]),

    ("HU-45", "planificador",
     "que al crear una tarea el sistema consulte al componente de estimacion y me muestre las horas previstas con su rango",
     "decidir con un margen de error conocido en lugar de un numero aislado",
     [(1, "Estimacion en frio",
       "Dado que escribo el titulo de una tarea sin indicar mi propia estimacion.",
       "Cuando la aplicacion consulta al componente de estimacion.",
       "Entonces muestra el valor probable en horas, su banda y el origen de la sugerencia."),
      (2, "Servicio no disponible",
       "Dado que el componente de estimacion no responde.",
       "Cuando escribo el titulo de la tarea.",
       "Entonces la aplicacion recurre al motor de estimacion del cliente y senala que la sugerencia proviene del respaldo."),
      (3, "Estimacion registrada",
       "Dado que el sistema me mostro una sugerencia.",
       "Cuando guardo la tarea.",
       "Entonces la estimacion queda registrada con sus horas, su banda, su origen y la version del modelo."),
      (4, "Cierre con horas reales",
       "Dado que una tarea estimada se marca como terminada con tiempo registrado.",
       "Cuando se cierra.",
       "Entonces las horas reales quedan en el historial del proyecto para el anclaje y el reentrenamiento.")]),

    ("HU-46", "planificador",
     "indicar mis propias estimaciones de horas y que el modelo las corrija a partir del texto de cada tarea",
     "conservar mi criterio sin renunciar a la evidencia",
     [(1, "Correccion de la estimacion del usuario",
       "Dado que escribo 6 horas como estimacion de una tarea.",
       "Cuando la guardo.",
       "Entonces el sistema conserva mi cifra y muestra la corregida, y el total del proyecto usa la corregida."),
      (2, "Cobertura del backlog",
       "Dado que he estimado menos del 80 % de las tareas del proyecto.",
       "Cuando consulto el total.",
       "Entonces el sistema indica que porcentaje llevo estimado y que el total mejora al completar el resto."),
      (3, "Origen visible",
       "Dado que el sistema devolvio una estimacion.",
       "Cuando consulto su detalle.",
       "Entonces veo si es en frio, anclada en tareas cerradas o corregida a partir de mi estimacion, y cuantas tareas cerradas o estimadas intervinieron."),
      (4, "Convivencia de las dos fuentes",
       "Dado que hay estimaciones mias y ademas tareas cerradas con horas reales.",
       "Cuando se calcula el total.",
       "Entonces el anclaje se contrae mas que en frio, de modo que anadir tareas cerradas nunca empeora el total.")]),

    ("HU-71", "jefe de proyecto",
     "ver el total de horas previsto del proyecto con su intervalo y que se recalcule al cerrar tareas",
     "saber cuanto falta con un margen de error conocido",
     [(1, "Plan provisional",
       "Dado un proyecto con menos de 3 tareas cerradas.",
       "Cuando abro su resumen.",
       "Entonces veo el total previsto marcado como provisional."),
      (2, "Total anclado",
       "Dado que el equipo cerro 3 tareas o mas con tiempo registrado.",
       "Cuando se recalcula el pronostico.",
       "Entonces el total suma las horas reales de las cerradas, ajusta las abiertas con el factor del proyecto e indica cuantas tareas lo anclan.")]),

    ("HU-72", "responsable del modelo",
     "que la contraccion del anclaje dependa de si el proyecto trae estimaciones del usuario",
     "que las dos palancas se sumen en lugar de estorbarse",
     [(1, "Contraccion en frio",
       "Dado un proyecto sin estimaciones del usuario.",
       "Cuando se ancla el total con k tareas cerradas.",
       "Entonces el factor de ajuste se contrae con k/(k+12)."),
      (2, "Contraccion con estimaciones",
       "Dado un proyecto con estimaciones del usuario ya corregidas.",
       "Cuando se ancla el total con k tareas cerradas.",
       "Entonces el factor se contrae con k/(k+96), porque tras la estimacion solo queda un 1,9 % de la varianza del residuo entre proyectos."),
      (3, "El anclaje no empeora",
       "Dado un proyecto con estimaciones del usuario y pocas tareas cerradas.",
       "Cuando se compara el total anclado con el no anclado.",
       "Entonces el anclado no es peor, cosa que si ocurria con la contraccion de 12.")]),

    ("HU-73", "jefe de equipo",
     "que la estimacion de una tarea tenga en cuenta quien la va a hacer",
     "repartir el trabajo sabiendo lo que cuesta a cada persona",
     [(1, "Ajuste por persona",
       "Dado que una persona del equipo tiene tareas cerradas con horas reales EN ESTE MISMO PROYECTO y se le asigna una tarea nueva.",
       "Cuando se estima esa tarea.",
       "Entonces sus horas se corrigen con el desplazamiento de esa persona, calculado como el cociente de sumas de sus cerradas contraido por k/(k+8)."),
      (2, "Sin historial de la persona",
       "Dado una persona sin tareas cerradas, o una tarea sin asignar.",
       "Cuando se estima.",
       "Entonces el ajuste es 1,0 y la estimacion queda como en frio."),
      (3, "Solo en la vista por tarea",
       "Dado un proyecto con ajustes por persona.",
       "Cuando se calcula el total del proyecto.",
       "Entonces el total NO los aplica, porque al sumar el proyecto entero se cancelan y empeoran el pronostico en 1,1 puntos sobre lo que queda abierto."),
      (4, "Ajuste visible",
       "Dado una tarea estimada con ajuste por persona.",
       "Cuando consulto su detalle.",
       "Entonces veo el factor aplicado y cuantas tareas cerradas de esa persona lo sostienen."),
      (5, "Es una funcion de equipo",
       "Dado un proyecto de una sola persona.",
       "Cuando se estima.",
       "Entonces el ajuste por persona coincide con el del proyecto y no aporta nada nuevo; su valor esta en equipos con ritmos distintos."),
      (6, "El historial de OTROS proyectos no cuenta",
       "Dado que esa persona tiene tareas cerradas en proyectos anteriores.",
       "Cuando se calcula su ajuste.",
       "Entonces esas tareas no entran: el desplazamiento de una misma persona correlaciona r=+0,089 entre sus proyectos, menos que el de dos personas DISTINTAS del mismo proyecto (r=+0,164).")]),

    ("HU-74", "planificador",
     "ver donde se concentra la probabilidad de una tarea, no solo un rango",
     "juzgar yo mismo si el riesgo me importa",
     [(1, "Tira de probabilidad",
       "Dado una tarea estimada.",
       "Cuando abro su detalle.",
       "Entonces veo los tramos de horas donde se concentra el 80 % de la probabilidad, cada uno con su peso."),
      (2, "Mas estrecha en medida, no en recorrido",
       "Dado la tira y la banda de la misma tarea.",
       "Cuando se comparan.",
       "Entonces la tira ocupa un 68 % menos de medida con la misma cobertura, pero abarca de punta a punta mas que la banda, porque se queda con los valores frecuentes y descarta los huecos."),
      (3, "No sustituye a la banda",
       "Dado que el usuario pide un rango unico.",
       "Cuando se muestra la estimacion.",
       "Entonces se usa la banda; la tira es una vista opcional del detalle."),
      (4, "Reparto mas parejo",
       "Dado el nivel nominal del 80 %.",
       "Cuando se mide la cobertura por tamano de tarea.",
       "Entonces la tira va de 0,45 a 0,86 y la banda de 0,25 a 1,00."),
      (5, "Recortar trozos rompe la promesa",
       "Dado que la interfaz quiere ensenar cinco tramos o menos.",
       "Cuando se recorta la tira.",
       "Entonces deja de cubrir el 80 % y es peor que la banda; el recorte es decision de presentacion y debe avisarse.")]),

    ("HU-48", "responsable del modelo",
     "armonizar las taxonomias y las unidades de los tres corpus en un unico dataset particionado",
     "que la evaluacion mida generalizacion y no memorizacion",
     [(1, "Particion cronologica por proyecto",
       "Dado el dataset armonizado.",
       "Cuando se generan las particiones.",
       "Entonces el bloque de prueba contiene las tareas mas recientes de cada proyecto y el bloque externo agrupa 150 proyectos enteros nunca vistos en entrenamiento."),
      (2, "Duplicados agrupados",
       "Dado dos tareas con el mismo hash de texto.",
       "Cuando se asignan a particiones.",
       "Entonces ambas caen en la misma particion."),
      (3, "Unidades unificadas",
       "Dado registros con esfuerzo en minutos, horas o puntos.",
       "Cuando se construye el dataset.",
       "Entonces el objetivo queda expresado en horas y las filas sin esfuerzo real se descartan."),
      (4, "Apache fuera del entrenamiento",
       "Dado el dataset armonizado.",
       "Cuando se selecciona el subconjunto del modelo.",
       "Entonces quedan 24.848 tareas de software de 204 proyectos y Apache se reserva como corpus de replica.")]),

    ("HU-49", "responsable del modelo",
     "representar el titulo de cada tarea con un codificador multilingue congelado y dos vistas lexicas",
     "dar a cada modelo la senal en la que acierta sin ajustar el codificador base",
     [(1, "Codificacion con cache",
       "Dado que parte de las tareas ya tiene su vector calculado.",
       "Cuando se ejecuta la codificacion.",
       "Entonces solo se codifica el texto nuevo y el resto se reutiliza."),
      (2, "Determinismo",
       "Dado el mismo texto codificado dos veces.",
       "Cuando se comparan los vectores.",
       "Entonces son identicos."),
      (3, "Solo el titulo",
       "Dado una tarea con titulo y descripcion.",
       "Cuando se codifica.",
       "Entonces solo interviene el titulo en ingles, que es lo que entrega el generador de tareas."),
      (4, "Rasgos prohibidos",
       "Dado atributos que no existen al crear la tarea, como la fuente, los enlaces, los componentes o los story points.",
       "Cuando se arman los rasgos.",
       "Entonces ninguno de ellos entra al modelo.")]),

    ("HU-64", "responsable del modelo",
     "dos copias ajustadas del codificador con objetivos distintos",
     "que una estime mejor la escala y la otra ordene mejor las tareas",
     [(1, "Dos objetivos distintos",
       "Dado el codificador base y el bloque de entrenamiento.",
       "Cuando se ajustan los brazos A y C.",
       "Entonces se guardan dos codificadores y sus vectores quedan alineados por fila con los del codificador base."),
      (2, "Codificador base intacto",
       "Dado el ajuste de los brazos A y C.",
       "Cuando termina el entrenamiento.",
       "Entonces los pesos del codificador base congelado no cambian y sus vectores en cache siguen siendo validos.")]),

    ("HU-50", "responsable del modelo",
     "un apilado que prediga la mediana de horas de una tarea a partir de su titulo",
     "mostrar cuanto tarda cada tarea cuando el proyecto todavia no tiene historial",
     [(1, "Metricas sobre prueba fija",
       "Dado el apilado ajustado en validacion.",
       "Cuando se evalua sobre las 4.225 tareas del bloque de prueba.",
       "Entonces se registran MAE 4,91 h, SA 0,431, MdMRE 0,670 y un 44,6 % de tareas dentro del factor 2."),
      (2, "Apilado ajustado en validacion",
       "Dado los cinco canales ya entrenados.",
       "Cuando se calculan los pesos del apilado.",
       "Entonces se usan las predicciones del bloque de validacion y no las de entrenamiento."),
      (3, "Generalizacion a proyectos nuevos",
       "Dado el bloque externo de proyectos nunca vistos.",
       "Cuando se evalua el mismo apilado.",
       "Entonces las metricas se reportan por separado sin reajustar el modelo."),
      (4, "La metrica no se reporta sola",
       "Dado que PRED(25) mide la dispersion y la curtosis de estimado/real, y que predecir la constante 2,45 h para TODAS las tareas saca un 17,8 % en el bloque de prueba, por encima de todo lo publicado.",
       "Cuando se informa el +-25 %.",
       "Entonces se acompana siempre del SA y del spearman, donde esa constante saca -0,015 y 0,000 y el modelo no."),
      (5, "Salida centrada",
       "Dado texto escrito por una persona.",
       "Cuando se piden las horas de una tarea.",
       "Entonces el sesgo mediano de la salida es x1,00; el factor de estilo del generador no se aplica a este texto, porque hacerlo lo dejaba en x0,82.")]),

    ("HU-66", "responsable del modelo",
     "estimar cada tarea por sus vecinas mas parecidas y llevar el resultado a los valores que la gente registra de verdad",
     "mostrar un valor probable reconocible en lugar de una cifra como 1,83 h",
     [(1, "Vecinos de acuerdo",
       "Dado una tarea cuyos 25 vecinos tienen horas parecidas.",
       "Cuando se estima.",
       "Entonces el valor probable es el valor de la rejilla mas cercano a la mediana de los vecinos."),
      (2, "Vecinos en desacuerdo",
       "Dado una tarea cuyos vecinos superan el umbral de dispersion.",
       "Cuando se estima.",
       "Entonces el valor probable es la prediccion del apilado."),
      (3, "Umbral elegido en validacion",
       "Dado el bloque de validacion.",
       "Cuando se barre el umbral de acuerdo.",
       "Entonces se conserva el que maximiza las tareas dentro del +-25 %."),
      (4, "El valor probable no se suma",
       "Dado un proyecto con sus tareas.",
       "Cuando se pide el total.",
       "Entonces el valor probable es solo para mostrar por tarea y el total NO lo suma."),
      (5, "La rejilla se aplica SIEMPRE",
       "Dado una tarea cuyos vecinos NO concuerdan y se cae a la prediccion del apilado.",
       "Cuando se calcula el valor probable.",
       "Entonces tambien se ajusta a la rejilla: hacerlo solo cuando concuerdan dejaba sin ajustar a la mayoria."),
      (6, "Tamano de la rejilla",
       "Dado que las horas registradas son un peine, donde treinta valores cubren la mitad del corpus.",
       "Cuando se elige la rejilla.",
       "Entonces son los 30 valores mas frecuentes, barridos fuera de pliegue: con 41 el +-25 % bajaba de 17,11 % a 16,59 %."),
      (7, "Lo que gana ajustar",
       "Dado el bloque de prueba.",
       "Cuando se compara la salida ajustada con la cruda.",
       "Entonces el +-25 % sube de 15,53 % a 17,02 % y el factor 2 de 0,403 a 0,424, sin mover el MAE, el SA ni el orden entre tareas.")]),

    ("HU-51", "responsable del modelo",
     "un modelo que reciba la estimacion del usuario y la prediccion en frio y devuelva las horas corregidas",
     "mejorar lo que la persona ya sabe en lugar de ignorarlo",
     [(1, "Mejora sobre la estimacion en frio",
       "Dado los 43 proyectos con estimacion del usuario.",
       "Cuando se aplica el corrector al total.",
       "Entonces los proyectos dentro del +-25 % suben de 44,2 % a 65,1 %."),
      (2, "Respaldo cuando falta la estimacion",
       "Dado una tarea sin estimacion del usuario.",
       "Cuando se calcula el total.",
       "Entonces esa tarea entra con su prediccion en frio."),
      (3, "Entrada del corrector",
       "Dado una tarea con estimacion del usuario.",
       "Cuando se corrige.",
       "Entonces el corrector recibe el vector del texto, el logaritmo de la estimacion y el logaritmo de la prediccion en frio, y se entrena fuera de pliegue agrupando por proyecto entero."),
      (4, "Estimar unas pocas tareas no basta",
       "Dado un proyecto en el que el usuario estimo 1, 3 o 5 tareas.",
       "Cuando se calcula el total.",
       "Entonces la mejora es marginal frente al frio; la ganancia completa exige estimar al menos el 80 % del backlog."),
      (5, "Control del artefacto de medicion",
       "Dado que en el 34,7 % de las tareas con estimacion el valor real es la estimacion copiada, porque la tarea se cierra imputando lo estimado.",
       "Cuando se repite la medida descartando esas filas.",
       "Entonces la ventaja por tarea aguanta: el +-25 % pasa de 15,2 % con solo texto a 23,8 % con la estimacion, y por proyecto de 46,2 % a 84,6 % en los 13 proyectos con menos del 25 % de filas identicas."),
      (6, "El modelo le gana a la estimacion cruda",
       "Dado el subconjunto sin filas identicas.",
       "Cuando se compara el corrector con la estimacion del usuario tal cual.",
       "Entonces el corrector acierta el 23,8 % frente al 21,8 % de la estimacion cruda; con las filas identicas dentro la comparacion se invierte y es enganosa.")]),

    ("HU-65", "responsable del modelo",
     "un segundo apilado que prediga la media de horas de cada tarea",
     "obtener el total del proyecto sumando valores que si se pueden sumar",
     [(1, "Total en frio frente a la linea base",
       "Dado los 114 proyectos de evaluacion predichos como si fueran nuevos.",
       "Cuando se suman las medias de sus tareas.",
       "Entonces el MdMRE del total es 0,355 frente a 0,694 de la linea base constante, con MAE 455 h y R2 en log 0,815."),
      (2, "Evaluacion por proyecto entero",
       "Dado el corpus del modelo.",
       "Cuando se arma la validacion cruzada.",
       "Entonces se agrupa por proyecto entero para que la escala del proyecto no se filtre al entrenamiento."),
      (3, "Las medianas no se suman",
       "Dado un proyecto con sus tareas.",
       "Cuando se pide el total.",
       "Entonces se suman las predicciones de la cabeza de la media y no las horas por tarea de la cabeza de la mediana."),
      (4, "Cancelacion al sumar",
       "Dado las mismas predicciones medidas por tarea y por proyecto.",
       "Cuando se comparan.",
       "Entonces el MdMRE pasa de 1,370 por tarea a 0,355 por proyecto, porque los errores se cancelan al sumar.")]),

    ("HU-52", "responsable del modelo",
     "recalcular el total del proyecto con las horas reales de las tareas ya cerradas y acompanarlo de un intervalo al 80 %",
     "que la interfaz muestre un margen de error honesto que mejora con el avance",
     [(1, "Anclaje con tareas cerradas",
       "Dado un proyecto con 5 tareas cerradas y sus horas reales.",
       "Cuando se recalcula el total.",
       "Entonces las cerradas entran con sus horas reales, las abiertas se ajustan con el factor del proyecto y el MdMRE baja de 0,355 a 0,292."),
      (2, "Contraccion con pocas tareas",
       "Dado un proyecto con una sola tarea cerrada.",
       "Cuando se ancla el total.",
       "Entonces el ajuste se contrae con k/(k+12) y el MdMRE queda en 0,346, por debajo del 0,355 en frio en lugar de empeorarlo."),
      (3, "Proyecto sin tareas cerradas",
       "Dado un proyecto sin tareas cerradas.",
       "Cuando se solicita el total.",
       "Entonces se devuelve el total en frio y la respuesta indica cero tareas cerradas."),
      (4, "Intervalo al 80 %, calibrado por tamano",
       "Dado el total estimado de un proyecto.",
       "Cuando se calcula su intervalo.",
       "Entonces la banda se calibra dentro de su tramo de tamano: x0,23 a x2,85 con 8-20 tareas, x0,17 a x2,13 con 21-60, x0,24 a x2,58 con 61-200 y x0,52 a x1,56 con mas de 200."),
      (5, "Formula del ajuste",
       "Dado k tareas cerradas con sus horas reales.",
       "Cuando se calcula el factor del proyecto.",
       "Entonces es el cociente de las sumas, log(suma real / suma predicha), contraido por k/(k+lambda), y se aplica solo a las tareas abiertas."),
      (6, "La cobertura se cumple en TODOS los tamanos",
       "Dado el nivel nominal del 80 %.",
       "Cuando se mide la cobertura por tramo de tamano.",
       "Entonces ningun tramo se desvia mas de 0,062 del 80 %, frente a 0,200 con una banda unica, que solo cubria el 67,7 % de los proyectos de 8-20 tareas."),
      (7, "Cuantil conforme, no muestral",
       "Dado m proyectos de calibracion.",
       "Cuando se toma el corte de la banda.",
       "Entonces se usa el estadistico de orden ceil((m+1)(1-a)) y no el cuantil muestral, que cubria el 78,1 % en vez del 80 % y el 74,6 % al calibrar por tramo.")]),

    ("HU-53", "responsable del modelo",
     "contrastar el estimador con LLM ejecutados localmente sobre las mismas tareas",
     "sustentar que el modelo propio es la opcion adecuada",
     [(1, "Veredicto pareado",
       "Dado las predicciones del estimador y de cada configuracion de LLM.",
       "Cuando se comparan sobre las mismas tareas.",
       "Entonces el modelo propio gana 17 comparaciones, empata 18 y pierde 1 de 36."),
      (2, "Costo de inferencia",
       "Dado el tiempo por tarea de ambos.",
       "Cuando se mide.",
       "Entonces el LLM tarda entre 0,31 y 2,45 s frente a 11 ms, entre 28 y 223 veces mas.")]),

    ("HU-54", "plataforma",
     "un servicio que reciba los titulos de las tareas y devuelva sus horas y el total del proyecto",
     "que la aplicacion consuma el modelo sin reimplementarlo",
     [(1, "Horas por tarea",
       "Dado una peticion con una lista de titulos.",
       "Cuando se llama a la estimacion por tarea.",
       "Entonces la respuesta trae horas, valor probable y banda de cada tarea."),
      (2, "Total del proyecto",
       "Dado una peticion con los titulos, las horas reales de las tareas cerradas y las estimaciones del usuario.",
       "Cuando se llama a la estimacion del proyecto.",
       "Entonces la respuesta trae el total, su banda al 80 %, el numero de tareas cerradas usadas y la contraccion aplicada."),
      (3, "Validacion de entrada",
       "Dado una peticion sin titulos.",
       "Cuando se envia.",
       "Entonces el servicio la rechaza con un error descriptivo sin invocar los modelos."),
      (4, "Servicio precalentado",
       "Dado el servicio recien arrancado.",
       "Cuando llega la primera peticion.",
       "Entonces los codificadores ya estan cargados y la respuesta no espera la carga.")]),

    ("HU-67", "responsable del modelo",
     "un paquete unico con las dos cabezas y un adaptador que reciba la salida del generador",
     "que el backlog generado salga con sus horas sin pasos manuales",
     [(1, "Tareas en ingles",
       "Dado un backlog generado con sus tareas en ingles y su traduccion al espanol.",
       "Cuando el adaptador pide las horas.",
       "Entonces envia las tareas en ingles y devuelve horas por tarea, subtotal por historia y total con intervalo."),
      (2, "Factor de estilo",
       "Dado tareas escritas al estilo del generador.",
       "Cuando se estima el total.",
       "Entonces se multiplica por 0,7273 antes del anclaje, porque el mismo modelo sobreestima x1,377 ese estilo frente a x1,002 del texto original de Jira."),
      (3, "Carga perezosa",
       "Dado un proceso que solo calcula el total del proyecto.",
       "Cuando estima un proyecto de 42 tareas.",
       "Entonces los dos codificadores ajustados no se cargan."),
      (4, "Prueba de extremo a extremo",
       "Dado un artefacto recien construido.",
       "Cuando se ejecuta la prueba final.",
       "Entonces estima texto nuevo con y sin tareas cerradas, y si falla el artefacto no se despliega."),
      (5, "Un factor por cabeza",
       "Dado las dos cabezas del paquete.",
       "Cuando se aplica el factor de estilo.",
       "Entonces la cabeza de la mediana usa 0,8313 y la de la media 0,7273, porque se midieron por separado.")]),

    ("HU-55", "responsable del modelo",
     "un trabajo por lotes que reajuste las cabezas con las tareas cerradas en la plataforma",
     "que la precision mejore con el uso sin tocar el codificador base",
     [(1, "Reajuste por lotes",
       "Dado tareas cerradas nuevas con horas reales.",
       "Cuando corre el trabajo.",
       "Entonces se escribe un artefacto nuevo versionado y el codificador base permanece intacto."),
      (2, "Sin datos nuevos",
       "Dado que no hay tareas cerradas desde el ultimo reajuste.",
       "Cuando corre el trabajo.",
       "Entonces termina sin sustituir el artefacto."),
      (3, "Artefacto verificado",
       "Dado un artefacto reentrenado.",
       "Cuando falla la prueba de extremo a extremo.",
       "Entonces no se publica y sigue en servicio el anterior."),
      (4, "Lo que el reajuste no promete",
       "Dado mas proyectos cerrados en la plataforma.",
       "Cuando se reajustan las cabezas.",
       "Entonces mejora el orden entre tareas, pero no se promete que el modelo aprenda a anticipar la escala de un proyecto nuevo: eso lo corrige el anclaje.")]),
]

REV = [
    ("HU-46", "REESCRITA",
     "'a partir del texto y del historial', y como criterio una sola tarea estimada.",
     "El historial sobre la estimacion del usuario no suma con lambda=12: empeora (70,5 % contra 73,2 %). Y estimar 1 tarea no mueve el total. Se separan las dos palancas y se anade el criterio de cobertura del backlog.",
     "E85 results/combinar_palancas/ · E84 results/cuantas_estimar/resumen.csv"),
    ("HU-52-2", "CIFRA FALSA",
     "'el MdMRE queda en 0,347 en lugar de 0,726 sin contraccion'.",
     "0,726 no es el proyecto sin contraer: es el MdMRE POR TAREA del bloque externo no Apache con cero anclas. No pinta nada ahi. El 0,347 si es correcto (0,3459). Se sustituye la comparacion por la unica medida: 0,346 contra 0,355 en frio.",
     "results/anclas_escala/resumen.csv k=1 · results/anclas/contraidas.csv fila 28"),
    ("HU-50-1", "CIFRAS CORREGIDAS",
     "'SA 0,432 y un 44,4 % de tareas dentro del factor 2'.",
     "Las dos estaban desviadas. El SA con cero anclas es 0,4309; el 0,4327 es la fila de 3 anclas. El factor 2 del modelo actual es 0,4464, es decir 44,6 % y no 44,4 %. Se anade ademas el MdMRE 0,670.",
     "results/final/resumen.csv · results/apilado_total/resumen.csv"),
    ("HU-35", "REESCRITA",
     "'veo las tareas mas parecidas de mi historico y su peso en el calculo'.",
     "Describe un mecanismo medido y descartado: ponderar el ajuste por similitud de texto pierde 0 de 5 comparaciones. El ajuste real es UN factor global del proyecto con todas las cerradas. Se separa la explicabilidad por tarea de la del total.",
     "E79 results/historial_por_texto/ · E80 results/residuo_es_global/"),
    ("HU-34", "AMPLIADA",
     "'mi propio historico', sin acotar el alcance.",
     "Se anade el criterio 2: los proyectos anteriores del mismo equipo NO entran. El sesgo es del proyecto, no del equipo (11,6 % entre organizaciones; los hermanos solo mejoran un 2,2 %), y pasar horas de hermanos resta 3,0 puntos.",
     "E87 results/sesgo_del_equipo/meta.json · E76 results/horas_del_equipo/"),
    ("HU-72", "NUEVA",
     "No existia ninguna HU para la contraccion condicional.",
     "Es el hallazgo de E86 y ya esta en el artefacto: lambda=12 en frio, lambda=96 con estimacion del usuario. Sin esto, anadir tareas cerradas sobre una estimacion empeora el total.",
     "E86 results/lambda_condicional/meta.json · src/171_estimador_v6.py"),
    ("HU-51-3/4", "ANADIDOS",
     "Faltaba que entra al corrector y cuanto backlog hay que estimar.",
     "El 3 fija la entrada y el protocolo fuera de pliegue por proyecto entero. El 4 evita prometer la mejora con pocas tareas estimadas.",
     "E84 results/cuantas_estimar/resumen.csv"),
    ("HU-51-5/6", "ANADIDOS · AMENAZA DE VALIDEZ",
     "Nada acotaba el artefacto de medicion de HoursEstimate.",
     "En el 34,7 % de las tareas con estimacion el real es la estimacion copiada (Spring 47,4 %, Jira 17,0 %, Sonatype 10,5 %). Sin esas filas la ventaja por tarea baja de +26,9 a +8,6 puntos pero aguanta, y por proyecto sube: 46,2 % a 84,6 % en los 13 proyectos limpios. Conviene declararlo nosotros antes que un revisor.",
     "E93 results/estimacion_igual_al_real/meta.json"),
    ("HU-65-1/4", "AMPLIADOS",
     "Solo daba el MdMRE.",
     "Se anaden MAE 455 h y R2log 0,815, y un criterio nuevo con la cancelacion al sumar: 1,370 por tarea contra 0,355 por proyecto. Es el argumento de por que el total se mide por proyecto.",
     "results/modelo_total/resumen.csv · results/anclas_escala/resumen.csv · E73 results/cancelacion/"),
    ("HU-66-4", "ANADIDO",
     "Nada impedia sumar el valor probable.",
     "La rejilla es solo para mostrar. Si alguien la suma, el total se hunde (queda en el 44 % del real). Conviene que sea criterio explicito.",
     "E73 results/cancelacion/"),
    ("HU-67-2/5", "PRECISADOS",
     "'la media por tarea queda en 6,3 h, cerca de las 6,11 h del corpus'.",
     "Esas dos cifras no estan en results/. Se sustituyen por lo medido: factor 0,7273 para la media y 0,8313 para la mediana, con los sesgos 1,002 y 1,377 que los originan.",
     "E74 results/estilo_total/"),
    ("HU-74", "NUEVA · SALIDA OPCIONAL",
     "La banda por tarea era lo unico que habia, y no se podia estrechar.",
     "E110-E112: la region de densidad predictiva maxima (HPD-split, Izbicki et al.) es un 68 % mas estrecha EN MEDIDA al 80 % (x10,4 contra x32,0) y reparte mucho mejor la cobertura por tramo. Sale en 43 trozos y su recorrido es mayor, asi que NO sustituye a la banda. Desplegada como tira_probabilidad(textos), con tau=0,9131 calibrado contra el bloque de validacion: cubre el 80,0 % con anchura mediana x7,09. El intervalo asimetrico, que la literatura sugiere para distribuciones sesgadas, no aporta aqui (x31,6 contra x32,0): el residuo en log es casi simetrico.",
     "E110 results/conjunto_mas_corto/ · E112 results/hpd_curva/ · src/230_parche_tira_hpd.py"),
    ("HU-73-6", "ANADIDO · ARRANQUE EN FRIO",
     "Nada impedia alimentar el ajuste con proyectos anteriores de la persona.",
     "E109 lo midio: pedir al usuario tareas pasadas de SUS desarrolladores para arrancar en caliente da SA 0,0287 contra 0,0299 en frio, por debajo incluso del control de tareas al azar (0,0272 es peor, pero ninguna gana). Solo las cerradas del PROPIO proyecto suben (0,0355). Y el contraste limpio: el desplazamiento de una misma persona correlaciona r=+0,089 entre sus proyectos, contra r=+0,164 entre personas distintas del mismo proyecto. El proyecto une mas que la persona.",
     "E109 results/arranque_con_historial/meta.json"),
    ("HU-73", "NUEVA",
     "No existia ninguna HU para el ajuste por persona.",
     "E100 descompuso el residuo en TRES niveles por primera vez: 12,4 % entre proyectos, 9,8 % entre desarrolladores del mismo proyecto y 77,8 % dentro del desarrollador. Ese 9,8 % estaba dentro del 89 % que E80 daba por ruido. Mejora el orden (spearman 0,317 a 0,358), el SA (0,026 a 0,036) y el MAE (5,06 a 5,00 h); el +-25 % solo sube 0,2 puntos porque la metrica premia concentrar. El criterio 3 lo deja FUERA del total, medido en E101.",
     "E100 results/anclaje_desarrollador/ · E101 results/persona_en_el_total/"),
    ("HU-52-5", "ANADIDO",
     "La formula del ajuste no estaba escrita en ningun sitio.",
     "Cociente de sumas contraido, aplicado solo a las abiertas. Es lo que distingue la implementacion de un promedio de cocientes.",
     "src/171_estimador_v6.py"),
    ("HU-55-4", "ANADIDO",
     "'la precision mejore con el uso', sin acotar en que.",
     "Mas proyectos de entrenamiento saturan el orden (Spearman 0,250 a 0,482) pero dejan la dispersion plana en 0,34 frente a 0,791 real. El reajuste no aprende la escala; eso lo hace el anclaje. Mejor no prometerlo.",
     "E81 results/curva_de_aprendizaje_escala/"),
    ("HU-52-4", "RESUELTA Y REESCRITA",
     "'la banda va de x0,36 a x2,01 del total'.",
     "Localizada: era la banda unica calculada con el cuantil muestral. Con la correccion conforme esa misma banda es x0,32 a x2,13, pero sobre todo deja de usarse: ahora se calibra por tramo de tamano. Se anaden los criterios 6 y 7 con la cobertura por tramo y el cuantil conforme.",
     "E89 results/banda_normalizada/resumen.csv · models/estimador_agix_v6.joblib"),
    ("HU-52-1/2", "SIN CAMBIO, VERIFICADOS DE NUEVO",
     "0,355 a 0,292 con 5 cerradas; 0,346 con 1.",
     "Siguen correctos. E88 confirma ademas que anclar en las tareas que de VERDAD se cerraron primero da +0,1 puntos frente al sorteo al azar, asi que estas cifras valen como medida de produccion.",
     "E88 results/orden_real_y_robustez/meta.json"),
    ("HU-52-5", "AMPLIADO",
     "La formula del ajuste.",
     "Se midio contra cuatro alternativas (media de logs, mediana de logs, Huber, sumas recortadas) y el cociente de sumas gana en todo k, con ventaja creciente: 54,8 % contra 43,1 % de Huber en k=20. Ponderar por tamano es lo correcto cuando el objetivo es una SUMA.",
     "E88 results/orden_real_y_robustez/resumen.csv"),
    ("HU-53", "VERIFICAR",
     "'gana 17, empata 18 y pierde 1 de 36' y '0,31 a 2,45 s frente a 11 ms'.",
     "No verificadas contra CSV en esta pasada.",
     "pendiente"),
    ("HU-63-2", "SIN TOCAR",
     "'cada tarea trae sus horas y el proyecto su total con intervalo al 80 %'.",
     "Es de la pipeline de generacion pero consume la de estimacion. Correcta tal cual; solo asegurate de que el adaptador aplique el factor de estilo de HU-67-2.",
     "ok"),
]


def main():
    w = openpyxl.Workbook()
    ws = w.active
    ws.title = "HU Estimacion"

    ws.merge_cells("A1:G1")
    ws["A1"] = "Enunciado de la Historia"
    ws.merge_cells("H1:M1")
    ws["H1"] = "Criterios de Aceptacion"
    cab = ["Identificador (ID) de la Historia", "Rol", "Caracteristica / Funcionalidad", "",
           "Razon / Resultado", "", "REDACCION", "Numero (#) de Escenario", "",
           "Criterio de Aceptacion (Titulo)", "Contexto", "Evento",
           "Resultado / Comportamiento esperado"]
    for j, v in enumerate(cab, 1):
        ws.cell(2, j, v)
    for j in range(1, 14):
        for i in (1, 2):
            c = ws.cell(i, j)
            c.font = Font(FUENTE, bold=True, size=10)
            c.fill = GRIS
            c.border = B
            c.alignment = Alignment("center", "center", wrap_text=True)

    r = 3
    for hid, rol, carac, razon, crits in HUS:
        r0 = r
        for n, tit, dado, cuando, ent in crits:
            ws.cell(r, 1, hid)
            if r == r0:
                ws.cell(r, 2, rol)
                ws.cell(r, 3, "Necesito:")
                ws.cell(r, 4, carac)
                ws.cell(r, 5, "Con la finalidad de:")
                ws.cell(r, 6, razon)
                ws.cell(r, 7, '=CONCATENATE("Como ",B%d," quiero ",D%d," para ",F%d)' % (r, r, r))
            ws.cell(r, 8, n)
            ws.cell(r, 9, '=CONCATENATE(A%d,"-",H%d)' % (r, r))
            ws.cell(r, 10, tit)
            ws.cell(r, 11, dado)
            ws.cell(r, 12, cuando)
            ws.cell(r, 13, ent)
            for j in range(1, 14):
                c = ws.cell(r, j)
                c.font = Font(FUENTE, size=10, bold=j in (3, 5))
                c.border = B
                c.alignment = Alignment("left", "center", wrap_text=True)
            ws.cell(r, 7).fill = VERDE
            ws.cell(r, 8).alignment = Alignment("center", "center")
            r += 1
        if len(crits) > 1:
            for j in (2, 3, 4, 5, 6, 7):
                ws.merge_cells(start_row=r0, start_column=j, end_row=r - 1, end_column=j)

    for col, anc in zip("ABCDEFGHIJKLM",
                        [14, 18, 11, 42, 18, 34, 46, 8, 11, 28, 40, 26, 62]):
        ws.column_dimensions[col].width = anc
    ws.freeze_panes = "A3"
    ws.row_dimensions[1].height = 18
    ws.row_dimensions[2].height = 32

    rv = w.create_sheet("Revision")
    rv.append(["HU", "Veredicto", "Lo que decia", "Por que se cambia", "Fuente verificada"])
    for j in range(1, 6):
        c = rv.cell(1, j)
        c.font = Font(FUENTE, bold=True, size=10)
        c.fill = GRIS
        c.border = B
        c.alignment = Alignment("center", "center", wrap_text=True)
    for i, fila in enumerate(REV, start=2):
        rv.append(list(fila))
        col = {"CIFRA FALSA": ROJO, "CIFRA SIN FUENTE": ROJO, "VERIFICAR": AMAR}.get(fila[1])
        for j in range(1, 6):
            c = rv.cell(i, j)
            c.font = Font(FUENTE, size=10, bold=j == 2)
            c.border = B
            c.alignment = Alignment("left", "top", wrap_text=True)
            if col and j == 2:
                c.fill = col
    for col, anc in zip("ABCDE", [12, 18, 50, 74, 44]):
        rv.column_dimensions[col].width = anc
    rv.freeze_panes = "A2"

    w.save(DESTINO)
    print("escrito", DESTINO)
    print(len(HUS), "HU ·", sum(len(h[4]) for h in HUS), "criterios ·",
          len(REV), "filas de revision")


if __name__ == "__main__":
    main()
