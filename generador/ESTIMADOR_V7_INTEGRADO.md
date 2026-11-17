# Estimador v7 integrado en producción

Aviso del chat del estimador (PBE-Dataset), 21/10/2026. Cambia el estimador que va dentro de
`F:\PBE-produccion` y el adaptador `src/34_estimacion.py` de **este** repositorio. La tubería de
generación no se ha tocado.

## Qué cambió

El estimador pasa de **v6** a **v7**: de **tres** codificadores e5-large (6,5 GB) a **uno**, en ONNX
int8 con el vocabulario podado (**397 MB**), que viaja dentro del estimador con su tokenizador. Ya
no hace falta `cache_hf` ni `HF_HOME`.

**Por qué.** Medido fuera de pliegue —GroupKFold por proyecto entero, que es el caso real de un
proyecto nuevo— los dos codificadores ajustados del v6 no aportaban; quitarlos **mejora** el orden
(Spearman 0,302 → 0,320) y el SA (0,0233 → 0,0268). En el bloque de prueba fijo el v6 parecía mejor
porque esos codificadores habían visto tareas de los mismos proyectos (la fuga ya conocida de E54).
int8 y poda no mueven las métricas del bloque de prueba. Detalle: `PBE-Dataset/src/246`–`254`.

## Archivos tocados

**En este repositorio**
- `src/34_estimacion.py`: busca `estimador_pbe_v7.joblib` y cae al v6 si no está;
  `hay_codificador_base()` mira el ONNX del estimador; docstrings al día. **Es el mismo archivo que
  el de producción**, copiado tal cual. Sin el paquete al lado, sigue tirando de PBE-Dataset, que
  ya tiene el v7.

**En `F:\PBE-produccion`**
- `generacion/estimador/models/`: fuera `e5_ajustado`, `e5_ajustado_C` y `estimador_pbe_v6.joblib`;
  dentro `estimador_pbe_v7.joblib` y `e5_v7/` (ONNX, tokenizador, tabla del vocabulario).
- `generacion/estimador/src/171_estimador_v6.py`: la clase sabe cargar el ONNX (sin torch ni red)
  y no carga los codificadores ajustados si el paquete no los trae. Con el v6 da **exactamente** lo
  mismo que antes (comprobado, desvío 0).
- `cache_hf/`: fuera, ya no lo usa nadie (solo lo usaba el estimador).
- `comprobar.py`: la lista nueva de archivos, `onnxruntime` entre los paquetes y `HF_HUB_OFFLINE=1`
  al cargar.
- `requisitos.txt`: `+ onnxruntime==1.31.0`.
- `INTEGRACION.md`: sin `HF_HOME`, tamaños y RAM nuevos, y una sección «El estimador v7» al final.

**Respaldo:** todo lo que salió del paquete —el v6, los dos codificadores, `cache_hf`, el
adaptador y los documentos anteriores— está en `F:\PBE-produccion_respaldo_v6`. Para volver al v6
basta con devolver esas carpetas a su sitio y el adaptador anterior.

## Verificado

| | v6 | v7 |
|---|---|---|
| paquete en disco | 6,0 GB | **3,1 GB** |
| RAM en CPU (pico, generar + estimar) | ~10 GB (doc.) | **5,2 GB** medido |
| estimar un backlog, en caliente, CPU | — | **~1 s** (la 1.ª llamada 5,7 s) |
| cargar el estimador | — | 1,7 s |

- `python comprobar.py --cargar` termina en **TODO FUNCIONA** sin red (`HF_HUB_OFFLINE=1`).
- Estimación por tarea con `HF_HOME` apuntando a una carpeta **vacía**: funciona, así que no
  depende de nada fuera del paquete.
- El total del proyecto y su banda salen **idénticos** al v6: los calcula la otra cabeza, que
  nunca usó los codificadores ajustados.

## Lo que hay que saber

1. **Las horas por tarea sobre texto del generador salen ~18 % más bajas que con el v6**
   (mediana sobre las 199 tareas de la prueba de extremo a extremo). El total no cambia. No se
   puede decir cuál de los dos acierta, porque las tareas generadas no tienen horas reales. El
   `factor_estilo` de la cabeza por tarea (0,8313) es el heredado; intenté re-medirlo con el corpus
   de 800 títulos reescritos, pero ese corpus lo reescribió un LLM local, no el generador, y el
   procedimiento no reproduce ni el valor del propio v6 (da 0,584). **Si en vuestro lado hay forma
   de producir tareas con el generador actual sobre tareas con horas reales conocidas**, eso
   cerraría la escala por tarea.
2. **`por_tarea=False` ya no ahorra memoria**: hay un solo codificador. Solo decide si se devuelven
   las horas de cada tarea.
3. **Una tarea que se sigue colando** en la prueba del paquete de producción: «Proceder para
   solicitudes recurrentes» (*Procedure for recurring requests*). El filtro nuevo de
   `tarea_degenerada` atrapa el verbo «proceed», pero esto es el **sustantivo** «Procedure for…».

## Si alguien reconstruye el paquete

Copiar desde PBE-Dataset: `models/estimador_pbe_v7.joblib`, `models/e5_v7/` entero y
`src/171_estimador_v6.py`. Las rutas del ONNX van **relativas al paquete** (la clase las resuelve
desde la carpeta que contiene `models/`), así que no hay que tocar nada más.
