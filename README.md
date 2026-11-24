# Space KO GTO

Fase 1 de un motor preflop 3-max para torneos Space KO. La arquitectura mantiene el evaluador de hojas detrás de una interfaz para poder sustituirlo por un solver postflop en una fase posterior.

## Estado

Este repositorio contiene los contratos de datos, equidad configurable, utilidad ICM/KO, un árbol público acotado, DCFR vectorial en PyTorch y persistencia/checkpoint SQLite. El recorrido propaga rangos completos `(3,1326)`; las hojas 3-way, las hojas de flop y la distribución de órdenes de showdown multiway siguen siendo aproximaciones.

El árbol público se puede construir desde Python con `build_tree((10, 10, 10))`; cada nodo expone estado, acciones con tamaños y ramas hijas. Las subidas se limitan a cuatro rondas y `build_tree` acepta `max_nodes` para evitar expansiones excesivas.

Para consultar SQLite desde Python, importa `BatchExporter` de `space_ko_gto.batch_exporter` y llama `combo_strategy(state_id, "ROOT", combo_index)`. `list_states()` y `list_node_paths(state_id)` permiten descubrir estados y nodos guardados.

## Inicio rápido

```bash
python -m pip install -e ".[equity,gpu]"
python -m space_ko_gto.equity_builder --output equity_1326x1326.npy
space-ko-solve --equity equity_1326x1326.npy --stacks 10 10 10 --iterations 1000
space-ko-batch --equity equity_1326x1326.npy --stack-buckets 2 3 4 --token-profiles EQUAL_LOW --prize-structures WTA --iterations 1000
```

El generador de equidad reutiliza cada board aleatorio para todas las manos compatibles (50.000 boards por defecto, unas 33.000 observaciones por pareja válida). `eval7` se instala como dependencia opcional. Los pares que comparten cartas quedan a cero y la matriz es simétrica por construcción. DCFR usa PyTorch y selecciona CUDA automáticamente cuando está disponible; el solver guarda checkpoints cada 100 iteraciones y exporta las estrategias una vez al terminar el estado. Usa `--resume` para continuar con la misma configuración. En WTA trabaja en ChipEV (bb); con varios puestos, evalúa ICM de resultados terminales y añade el valor KO solo al bustear. `--token-scale` escala el componente KO para que esté en las mismas unidades que el solver (`--payoff-to-bb` sigue aceptado como alias). `--nashconv-every 200` calcula NashConv determinista por rango y mejor respuesta por combo; la métrica corresponde al modelo terminal aproximado descrito arriba, no a un solver postflop exacto.

`space-ko-batch` recorre la rejilla seleccionada secuencialmente; limita cada ejecución con `--limit` y continúa estados interrumpidos con `--resume`. Los estados fallidos se marcan `FAILED` y su causa queda en `state_errors`. La tabla `state_metadata` conserva stacks, tokens, premios, factor de unidades y hash de la matriz de equidad para cada `state_id`.

## Contratos

- Combos: orden lexicográfico de `(c1, c2)` con IDs de carta `rank * 4 + suit`.
- Posiciones: BTN, SB, BB; ciegas 0, 0.5 y 1 bb.
- Equidades: `float32`, forma `(1326, 1326)`.
- Persistencia: SQLite portable con colas de estados y estrategias comprimidas `float16`.

`VectorDCFRSolver` (también disponible con el nombre compatible `MCCFRSolver`) acepta un `BaseLeafEvaluator` basado en rangos completos. Su método `checkpoint_fingerprint()` identifica el evaluador para impedir reanudaciones incompatibles. Consulta `docs/PHASE2_POSTFLOP_SPEC.md` para el contrato de extensión postflop.
