# Benchmark protocols and metrics

- HiddenBench translates `agent_id`, `round_num`, and content. Its grader owns
  pre/post accuracy, empirical information gain, and consensus.
- COMMA translates role/turn/modality fields. Its grader owns completion and
  Telehealth partial credit.
- Collab-Overcooked has distinct episode and timestep adapters. Its grader owns
  success, TES/recall, F1, similarity, redundancy, and collaboration.
- iAgents translates sender/receiver conversations. Its grader owns normalized
  answer matching and an optional independent regrade.

All adapters report canonical raw and transmitted byte/message counts. Smoke
fixtures require no GUI, database, network API, or live model server.

