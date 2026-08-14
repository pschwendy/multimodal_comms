# Benchmark protocols and metrics

- [HiddenBench](../../src/multimodal_comms/benchmarks/hiddenbench/README.md)
  translates `agent_id`, `round_num`, and content. Its grader owns pre/post
  accuracy, empirical information gain, and consensus.
- [COMMA](../../src/multimodal_comms/benchmarks/comma/README.md) translates
  role/turn/modality fields. Its grader owns completion and Telehealth partial
  credit.
- [Collab-Overcooked](../../src/multimodal_comms/benchmarks/collab_overcooked/README.md)
  has distinct episode and timestep adapters. Its grader owns success,
  TES/recall, F1, similarity, redundancy, and collaboration.
- [iAgents](../../src/multimodal_comms/benchmarks/iagents/README.md) translates
  sender/receiver conversations. Its grader owns normalized answer matching and
  an optional independent regrade.

All adapters report canonical raw and transmitted byte/message counts. Smoke
fixtures require no GUI, database, network API, or live model server.
