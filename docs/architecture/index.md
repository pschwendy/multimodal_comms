# Architecture and boundaries

The dependency direction is deliberate:

1. `core` defines immutable messages, context, transmission accounting, and
   structural codec/packer protocols.
2. `methods` depends only on `core` and optional numerical/model libraries.
3. `benchmarks` translates native records and owns native grading.
4. `evaluation`, `training`, `apps`, and `cli` consume those public layers.

`full_history` and `delta` are channel-view policies, not compressors. A view
chooses which messages are eligible; a `CommunicationMethod` transforms that
view. This removes protocol conditionals from method implementations.

Model/tokenizer/representation access is injected. Importing a method never
downloads a model, selects a CUDA device, contacts a server, or reads a path
outside the repository root.
