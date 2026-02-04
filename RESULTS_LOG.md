\# Experiment Findings (1M Scale)



\## Issue: Semantic Drift

At 1 million documents, 'bert-tiny' struggled to distinguish between entities in the same domain.

\- \*\*Query:\*\* 'apple financial revenue'

\- \*\*Result:\*\* Cisco Systems (Score 0.37) vs Apple (Score 0.38) - Too close / False Positives.



\## Solution: Hybrid Weight Tuning

Pure vector search is insufficient for entity-specific queries. We are implementing a weighted alpha parameter to balance Semantic Understanding (Vectors) with Entity Precision (Keywords).

