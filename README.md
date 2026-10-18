# HNSW-Based Vector Similarity Search in C

An individual Data Structures and Algorithms mini-project implementing
Hierarchical Navigable Small World (HNSW) graph-based approximate
nearest-neighbour search in C.

## Project overview

The project contains a layered graph index, vector and node structures,
priority heaps, neighbour-selection and pruning operations, and a
search implementation.

An exact brute-force baseline is included for comparison and Recall@k
evaluation.

The semantic-search demonstration uses precomputed sentence embeddings
generated using the pretrained sentence-transformers/all-MiniLM-L6-v2
model. The language model is not trained by this project. Its embedding
generation scripts are not included in this C-only submission.

## Repository structure

- `include/`: C header files and data-structure interfaces
- `src/`: HNSW, node, vector, heap and exact-search implementations
- `examples/`: Semantic-search demonstration
- `benchmarks/`: Synthetic benchmark
- `data/`: Corpus text and precomputed sentence embeddings
- `Makefile`: Build targets for the semantic demo and benchmark

## Requirements

- GCC or a compatible C compiler
- GNU Make
- A C math library compatible with `-lm`

## Build the semantic-search executable

```bash
make semantic-search
