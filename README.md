# Exploring the characteristics of PLM-based knowledge graph embeddings

Repository for the course project of 263-5000-00L Computational Semantics for Natural Language Processing FS2022 at ETH.

Code and data for experiments in section 5 can be found under [extrapolation](extrapolation/).

Code and data for experiments in section 4 and 6 can be found under [embedding_distribution](embedding_distribution/).

Results for ablation models are under [ablation](ablation/).

Other branches in the repo are our attempts to improve SimKGC. Most of these results are not ideal and thus are not covered in the report. A brief overview of these branches:

- `similarity-metrics`: Replacing cosine similarity with dot product or euclidean distance.
- `negsamples`: Attempt to design a commonsense-aware negative sampler similar to CAKE.
- `cake-loss` and `loss-func`: Attempt to design a commonsense-aware or a relation type-aware loss function.
- `ensemble`: Reverse the hr + t structure of SimKGC to investigate the backward performance.
- `adapter`: Incorporating concept prediction task through AdapterFusion.
