# Space KO GTO

Fase 1 de un motor preflop 3-max para torneos Space KO. La arquitectura mantiene el evaluador de hojas detrás de una interfaz para poder sustituirlo por un solver postflop en una fase posterior.

## Estado

Este repositorio contiene los contratos de datos, cálculo de equidad configurable, utilidad ICM/KO, rejilla de estados, árbol público acotado, MCCFR de muestreo externo y persistencia/checkpoint SQLite. Las hojas heads-up usan equidad preflop; las hojas 3-way y los nodos que llegan al flop usan aproximaciones. El motor aún no ofrece un cálculo validado de exploitabilidad/NashConv y sus estrategias no deben tratarse como una solución GTO exacta.

El árbol público se puede construir desde Python con `build_tree((10, 10, 10))`; cada nodo expone estado, acciones con tamaños y ramas hijas. Las subidas se limitan a cuatro rondas y `build_tree` acepta `max_nodes` para evitar expansiones excesivas.

Para consultar SQLite desde Python, importa `BatchExporter` de `space_ko_gto.batch_exporter` y llama `combo_strategy(state_id, "ROOT", combo_index)`. `list_states()` y `list_node_paths(state_id)` permiten descubrir estados y nodos guardados.

## Inicio rápido

```bash
python -m pip install -e ".[equity]"
python -m space_ko_gto.equity_builder --output equity_1326x1326.npy --trials 10000
space-ko-solve --equity equity_1326x1326.npy --stacks 10 10 10 --iterations 1000
space-ko-batch --equity equity_1326x1326.npy --stack-buckets 2 3 4 --token-profiles EQUAL_LOW --prize-structures WTA --iterations 1000
```

El generador de equidad reutiliza cada board aleatorio para todas las manos compatibles (10.000 boards por defecto, unas 6.600 observaciones por pareja válida). `eval7` se instala como dependencia opcional. Los pares que comparten cartas quedan a cero y la matriz es simétrica por construcción. El solver guarda las medias y su estado de ejecución cada 100 iteraciones; usa `--resume` para continuar un checkpoint con la misma configuración. `--payoff-to-bb` convierte los valores KO expresados en unidades de premio a la escala bb del solver; el valor por defecto `1.0` requiere una calibración externa. `TorchPreflopEQRLeafEvaluator` puede ejecutar su kernel en CUDA. El recorrido MCCFR sigue en CPU. `--exploitability-samples 100` calcula una cota superior de exploitabilidad “clairvoyant” por muestreo: su mejor respuesta observa las cartas rivales, así que no es NashConv de información incompleta.

`space-ko-batch` recorre la rejilla seleccionada secuencialmente; limita cada ejecución con `--limit` y continúa estados interrumpidos con `--resume`. Los estados fallidos se marcan `FAILED` y su causa queda en `state_errors`.

## Contratos

- Combos: orden lexicográfico de `(c1, c2)` con IDs de carta `rank * 4 + suit`.
- Posiciones: BTN, SB, BB; ciegas 0, 0.5 y 1 bb.
- Equidades: `float32`, forma `(1326, 1326)`.
- Persistencia: SQLite portable con colas de estados y estrategias comprimidas `float16`.

El constructor `MCCFRSolver(..., leaf_evaluator=...)` acepta una implementación de `BaseLeafEvaluator`; su método `checkpoint_fingerprint()` identifica la configuración para no reanudar con un evaluador distinto. Si ofrece `evaluate_sampled_leaf`, el solver la usa para la hoja muestreada; en caso contrario llama `evaluate_postflop_leaf` con reaches one-hot. Consulta `docs/PHASE2_POSTFLOP_SPEC.md` para el contrato de extensión postflop.
