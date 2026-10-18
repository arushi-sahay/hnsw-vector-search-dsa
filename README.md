# HNSW-Based Vector Similarity Search in C

**Data Structures Mini-Project**

An implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate nearest-neighbour search, evaluated against exact brute-force search and demonstrated with sentence embeddings.

## 1. Project Overview

Given a query vector, the system retrieves the nearest vectors from an indexed collection.

The project implements the HNSW graph index and its supporting data structures in C. It includes two demonstrations:

1. **Semantic Search:** Retrieves sentences using precomputed, 384-dimensional sentence embeddings.
2. **Synthetic Benchmark:** Compares HNSW with exact brute-force search across multiple dataset sizes and search configurations.

The pretrained `sentence-transformers/all-MiniLM-L6-v2` model was used to generate the supplied embeddings. This project does not train or fine-tune a language model. The embedding-generation Python scripts are not part of this C-only submission.

## 2. Quick Start: Build and Run Everything

### Requirements

- Git
- GCC
- GNU Make
- A C math library compatible with `-lm`

On Fedora, install the required build tools if necessary:

```bash
sudo dnf install gcc make git
```

### Run the Entire Demonstration

Copy and paste this single command into a Linux terminal:

```bash
git clone https://github.com/arushi-sahay/hnsw-vector-search-dsa.git && cd hnsw-vector-search-dsa && make clean && make demo
```

This command:

1. Downloads the source code and supplied data.
2. Compiles the semantic-search program with `DIM=384`.
3. Compiles the synthetic benchmark with `DIM=64`.
4. Runs the semantic-search example and compares its results against exact brute-force search.
5. Runs the full synthetic benchmark and prints the measurements as CSV.

No Python environment, GPU, model download, or external machine-learning library installation is required to run the supplied C demonstrations.

If the repository has already been cloned, run the following command from the repository directory:

```bash
make clean && make demo
```

## 3. What the Programs Demonstrate

### A. Semantic Search

The semantic-search program loads the supplied sentence embeddings and their corresponding original sentences, inserts the vectors into an HNSW index, and searches for the nearest neighbours of the supplied query vector.

The included query represents:

> What causes solar storms and coronal mass ejections?

The corpus contains 48 sentences. Each sentence is represented by a 384-dimensional, L2-normalized vector generated using the same pretrained embedding model.

The program displays:

- The nearest retrieved sentences
- Euclidean distances
- Cosine similarities
- Exact brute-force results
- Recall@k

In the recorded example, HNSW returned the same five results as brute-force search, giving Recall@5 = 5/5 = 100%.

**Important interpretation:** Recall@5 measures agreement with exact nearest-neighbour search. It does not, by itself, establish that every returned result is semantically relevant to a human reader.

The query vector is supplied as a data file. This C-only repository does not generate embeddings for arbitrary new text.

### B. Synthetic Benchmark

The benchmark generates synthetic clustered vectors and compares HNSW against exact brute-force search.

The experimental configuration is:

- Vector dimensions: 64
- Dataset sizes: 250, 750, and 1,500 vectors
- Number of clusters: 8
- Number of requested neighbours: k = 5
- Graph connectivity parameter: M = 8 and 16
- Construction parameter: ef_construction = 100
- Search parameter: ef_search = 16, 64, and 128
- Queries per configuration: 40

The benchmark prints CSV rows containing:

- Graph construction time
- Mean brute-force query latency
- Mean HNSW query latency
- Reciprocal-of-mean latency QPS
- Recall@5
- Directed adjacency-entry count
- Estimated index memory

### Understanding the Results

HNSW is an approximate search algorithm. The search parameter `ef_search` controls the breadth of candidate exploration. A higher value can improve recall but usually increases query latency.

The parameter `M` influences graph connectivity. Increasing it can increase construction cost, adjacency storage, and memory usage.

Benchmark timings depend on the machine and system load, so exact timing values can vary between runs. The reported memory is an index estimate, not total process memory usage.

These experiments provide evidence from a small synthetic benchmark. They do not guarantee the same performance at every scale or on every dataset.

## 4. Data Structures and Source Files

| Path | Responsibility |
|---|---|
| `include/hnsw.h`, `src/hnsw.c` | HNSW index, insertion, layered search, neighbour selection, and pruning |
| `include/node.h`, `src/node.c` | Graph nodes and per-layer neighbour lists |
| `include/vector.h`, `src/vector.c` | Fixed-dimensional vectors and Euclidean distance |
| `include/heap.h`, `src/heap.c` | Min-heap and max-heap priority queues |
| `include/bruteforce.h`, `src/bruteforce.c` | Exact nearest-neighbour baseline |
| `examples/semantic_search.c` | Sentence retrieval and Recall@k comparison |
| `benchmarks/benchmark_hnsw.c` | Synthetic performance and recall benchmark |
| `data/corpus.txt` | Source corpus of sentences |
| `data/semantic/embeddings.csv` | Precomputed sentence vectors |
| `data/semantic/sentences.txt` | Original sentences aligned by row with the vectors |
| `data/semantic/query_embedding.csv` | Precomputed query vector for the supplied semantic demo |
| `data/semantic/manifest.json` | Embedding metadata and row-mapping information |
| `Makefile` | Build, run, and clean commands |

## 5. How HNSW Works

HNSW stands for **Hierarchical Navigable Small World**. It is a graph-based data structure designed to make nearest-neighbour retrieval more efficient.

Each node stores a vector and maintains neighbour lists at the graph layers where it exists.

- **Layer 0:** Contains all indexed nodes and supports detailed local exploration.
- **Upper layers:** Contain fewer nodes and provide longer-range navigation.
- **Insertion:** Assigns a maximum layer to a node, searches for suitable neighbours, establishes connections, and prunes lists that exceed their allowed limits.
- **Query search:** Starts from the entry point at the highest layer, moves towards closer nodes, descends through the hierarchy, and performs a broader best-first search at Layer 0.

The implementation uses priority heaps to manage candidates and retained results. A visited array prevents repeatedly processing the same node during a layer search.

HNSW does not connect every possible pair of vectors. Instead, it constructs a navigable graph with selected connections, balancing search quality and graph size.

## 6. Useful Commands

### Build Only the Semantic-Search Executable

```bash
make semantic-search
```

### Run the Supplied Semantic Query

```bash
./build/semantic_search data/semantic/embeddings.csv data/semantic/sentences.txt data/semantic/query_embedding.csv 5 32
```

The final arguments specify `k=5` and `ef_search=32`.

### Build and Run Only the Synthetic Benchmark

```bash
make benchmark
```

### Run Both Demonstrations

```bash
make demo
```

### Remove Compiled Files

```bash
make clean
```

## 7. Evaluation Metrics

### Query Latency

Query latency is the average time required to execute one query. Lower latency means faster retrieval under the measured experimental conditions.

### Recall@5

Recall@5 measures how many of the five exact nearest-neighbour results are also present in the HNSW result set, divided by five.

For example, if HNSW retrieves four of the five exact nearest neighbours:

Recall@5 = 4/5 = 80%

A result of 100% means the two methods returned the same top-five set for the evaluated query. It does not measure human-rated relevance.

### Queries per Second (QPS)

The benchmark reports the reciprocal of mean query latency. This is a derived latency-based rate, not a measurement of concurrent request throughput.

### Estimated Index Memory

The memory estimate covers the index structure, allocated node-pointer capacity, node structures, neighbour-list arrays, and their ID capacities. It does not include allocator metadata, external corpus and query buffers, temporary search heaps, or total process memory usage.

## 8. Scope and Limitations

- The graph index, vector operations, heaps, search, and brute-force baseline are implemented in C.
- Sentence embeddings are supplied as precomputed data. Embedding generation is outside this C-only submission.
- The semantic corpus contains only 48 sentences, and the demonstration uses a fixed query vector.
- Recall measures agreement with exact nearest-neighbour search rather than human-rated semantic quality.
- The synthetic benchmark uses relatively small datasets and a limited number of queries.
- Timing measurements can vary across runs and environments.
- The experiments do not establish universal performance guarantees or production-scale suitability.

The project's main contribution is a working C implementation of HNSW, together with an exact-search baseline, a synthetic evaluation, and a small semantic-search demonstration using pretrained sentence embeddings.

## 9. References

1. Malkov, Y. A., and Yashunin, D. (2020). "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836.  
   https://arxiv.org/abs/1603.09320

2. Sentence Transformers. `all-MiniLM-L6-v2` model card.  
   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

3. hnswlib. Algorithm parameters: M, ef_construction, and ef search.  
   https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
