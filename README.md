# Space KO GTO

Fase 1 de un motor preflop 3-max para torneos Space KO. La arquitectura mantiene el evaluador de hojas detrás de una interfaz para poder sustituirlo por un solver postflop en una fase posterior.

## Estado

Este repositorio contiene los contratos de datos, generación determinista de combos, cálculo de equidad configurable, utilidad ICM/KO, rejilla de estados, evaluador de hojas y utilidades de persistencia SQLite. El árbol de apuestas y el solver completo DCFR aún necesitan implementación y validación antes de producir estrategias publicables. Las cifras EQR de la especificación son heurísticas iniciales, no una solución postflop exacta.

## Inicio rápido

```bash
python -m pip install -e ".[equity]"
python -m space_ko_gto.equity_builder --output equity_1326x1326.npy --trials 10000
```

El generador de equidad usa Monte Carlo (10.000 boards por defecto) para cada pareja válida, lo que puede requerir mucho tiempo. `eval7` se instala como dependencia opcional. Los pares que comparten cartas quedan a cero y la matriz es simétrica por construcción.

## Contratos

- Combos: orden lexicográfico de `(c1, c2)` con IDs de carta `rank * 4 + suit`.
- Posiciones: BTN, SB, BB; ciegas 0, 0.5 y 1 bb.
- Equidades: `float32`, forma `(1326, 1326)`.
- Persistencia: SQLite portable con colas de estados y estrategias comprimidas `float16`.

Consulta `docs/PHASE2_POSTFLOP_SPEC.md` para el contrato de extensión postflop.
