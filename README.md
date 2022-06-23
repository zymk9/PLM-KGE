# CocaKE
Contrastive Ontology and Commonsense-Aware Knowledge Embedding

## Metrics


## Per hr temperature

| Model                           | Forward |        |        |        | Backward |        |        |        |
|---------------------------------|:-------:|:------:|:------:|:------:|:--------:|:------:|:------:|:------:|
|                                 |  1-to-1 | 1-to-n | n-to-1 | n-to-n |  1-to-1  | 1-to-n | n-to-1 | n-to-n |
| Learned t                       |  0.0406 | 0.0549 | 0.0477 | 0.0578 |  0.0424  | 0.0450 | 0.0519 | 0.0574 |
| Learned t (inverse only)        |  0.0278 | 0.0311 | 0.0392 | 0.0582 |  0.0267  | 0.0379 | 0.0333 | 0.0605 |
| Learned t (with type pred task) |  0.0569 | 0.0767 | 0.0636 | 0.2601 |  0.0586  | 0.0746 | 0.0618 | 0.2775 |