# Agix v7 frente a LLM

601 tareas con horas reales (301 del bloque de prueba y 300 de proyectos externos), ninguna vista por el v7 al entrenar. Generado por `src/261_tablas_llm_api.py`.

## Condición A · solo el texto de la tarea (como en producción)

| modelo | proveedor | versión | n | válidas | MdMRE | PRED(25) | MAE (h) | SA | Spearman | frente a Agix | s/tarea |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Agix v7 (nuestro)** | — | — | 601 | 100 % | 0,794 | 16,5 % | 4,23 | 0,408 | 0,246 | — | 0,0137 |
| **Agix v7 · valor probable** | — | — | 601 | 100 % | 0,769 | 15,1 % | 4,31 | 0,396 | 0,198 | — | 0,0137 |
| **constante: mediana global** | — | — | 601 | 100 % | 0,750 | 17,1 % | 4,27 | 0,402 | — | — | — |
| GPT-OSS 120B (OpenAI) | groq | calibrado | 601 | 96,0 % | 0,750 | 18,8 % | 4,26 | 0,404 | 0,323 | empate [-0.140, +0.001] | 3,93 |
| GPT-OSS 120B (OpenAI) | groq | crudo | 601 | 96,0 % | 3,200 | 8,7 % | 9,35 | -0,308 | 0,323 | Agix mejor [+1.938, +2.751] | 3,93 |
| GPT-OSS 20B (OpenAI) | groq | calibrado | 601 | 95,0 % | 0,718 | 14,1 % | 4,28 | 0,401 | 0,299 | LLM mejor [-0.170, -0.021] | 4,53 |
| GPT-OSS 20B (OpenAI) | groq | crudo | 601 | 95,0 % | 5,000 | 7,7 % | 12,72 | -0,780 | 0,299 | Agix mejor [+3.311, +4.777] | 4,53 |
| Qwen 3.8 27B | groq | calibrado | 601 | 100,0 % | 0,739 | 15,8 % | 4,39 | 0,386 | 0,315 | LLM mejor [-0.145, -0.002] | 2,50 |
| Qwen 3.8 27B | groq | crudo | 601 | 100,0 % | 2,200 | 14,3 % | 7,68 | -0,074 | 0,315 | Agix mejor [+0.976, +2.181] | 2,50 |
| Gemma 2 9B | local (Ollama) | calibrado | 601 | 100,0 % | 0,743 | 10,6 % | 4,19 | 0,414 | 0,333 | LLM mejor [-0.141, -0.000] | 0,84 |
| Gemma 2 9B | local (Ollama) | crudo | 601 | 100,0 % | 2,429 | 10,3 % | 7,12 | 0,004 | 0,333 | Agix mejor [+1.188, +2.206] | 0,84 |
| Llama 3.1 8B | local (Ollama) | calibrado | 601 | 100,0 % | 0,750 | 13,6 % | 4,22 | 0,410 | 0,296 | empate [-0.119, +0.030] | 2,40 |
| Llama 3.1 8B | local (Ollama) | crudo | 601 | 100,0 % | 3,800 | 8,7 % | 9,71 | -0,359 | 0,296 | Agix mejor [+2.226, +3.572] | 2,40 |
| Mistral 7B | local (Ollama) | calibrado | 601 | 100,0 % | 0,822 | 12,8 % | 4,79 | 0,330 | 0,283 | empate [-0.068, +0.084] | 0,32 |
| Mistral 7B | local (Ollama) | crudo | 601 | 100,0 % | 7,000 | 4,8 % | 20,77 | -1,905 | 0,283 | Agix mejor [+5.478, +7.788] | 0,32 |
| Qwen 2.5 7B | local (Ollama) | calibrado | 601 | 100,0 % | 0,762 | 12,5 % | 4,14 | 0,420 | 0,335 | empate [-0.119, +0.019] | 0,31 |
| Qwen 2.5 7B | local (Ollama) | crudo | 601 | 100,0 % | 2,000 | 11,0 % | 5,35 | 0,251 | 0,335 | Agix mejor [+0.734, +1.377] | 0,31 |

## Condición B · el LLM ve además ejemplos con horas reales

| modelo | proveedor | versión | n | válidas | MdMRE | PRED(25) | MAE (h) | SA | Spearman | frente a Agix | s/tarea |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Agix v7 (nuestro)** | — | — | 601 | 100 % | 0,794 | 16,5 % | 4,23 | 0,408 | 0,246 | — | 0,0137 |
| **Agix v7 · valor probable** | — | — | 601 | 100 % | 0,769 | 15,1 % | 4,31 | 0,396 | 0,198 | — | 0,0137 |
| **constante: mediana global** | — | — | 601 | 100 % | 0,750 | 17,1 % | 4,27 | 0,402 | — | — | — |
| Qwen 3.8 27B | groq | crudo | 218 | 69,3 % | 0,667 | 18,8 % | 3,81 | 0,419 | 0,249 | empate [-0.224, +0.020] | 181,30 |
| Gemma 2 9B | local (Ollama) | calibrado | 601 | 100,0 % | 0,739 | 12,6 % | 4,13 | 0,422 | 0,325 | LLM mejor [-0.143, -0.003] | 1,05 |
| Gemma 2 9B | local (Ollama) | crudo | 601 | 100,0 % | 0,700 | 15,3 % | 4,13 | 0,423 | 0,325 | LLM mejor [-0.190, -0.026] | 1,05 |
| Llama 3.1 8B | local (Ollama) | calibrado | 601 | 100,0 % | 0,800 | 16,8 % | 4,24 | 0,406 | 0,295 | empate [-0.082, +0.041] | 2,45 |
| Llama 3.1 8B | local (Ollama) | crudo | 601 | 100,0 % | 0,750 | 16,3 % | 4,17 | 0,416 | 0,295 | LLM mejor [-0.138, -0.001] | 2,45 |
| Mistral 7B | local (Ollama) | calibrado | 601 | 100,0 % | 0,813 | 13,1 % | 4,55 | 0,364 | 0,328 | empate [-0.075, +0.074] | 0,38 |
| Mistral 7B | local (Ollama) | crudo | 601 | 100,0 % | 0,806 | 14,3 % | 4,44 | 0,379 | 0,328 | empate [-0.078, +0.072] | 0,38 |
| Qwen 2.5 7B | local (Ollama) | calibrado | 601 | 100,0 % | 0,699 | 17,5 % | 4,28 | 0,401 | 0,345 | LLM mejor [-0.186, -0.022] | 0,39 |
| Qwen 2.5 7B | local (Ollama) | crudo | 601 | 100,0 % | 0,714 | 16,1 % | 4,10 | 0,426 | 0,345 | LLM mejor [-0.188, -0.018] | 0,39 |

**Lectura.** «frente a Agix» es el veredicto sobre el IC95 de la diferencia de MdMRE (LLM − Agix) en las mismas tareas: negativo y sin cruzar el cero, gana el LLM; positivo, gana Agix. Las respuestas inválidas del LLM cuentan como la mediana global. «calibrado» multiplica al LLM por un factor aprendido en 130 tareas aparte. Agix responde en 0,0137 s por tarea en CPU, sin conexión y sin coste por llamada.
