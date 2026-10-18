CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wpedantic
CPPFLAGS = -Iinclude
LDLIBS = -lm

BUILD_DIR = build
SEMANTIC_DIM = 384
BENCH_DIM = 64

CORE_SRC = src/hnsw.c src/node.c src/vector.c src/heap.c src/bruteforce.c

.PHONY: all semantic-search benchmark clean

all: semantic-search

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/semantic_search: examples/semantic_search.c $(CORE_SRC) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DDIM=$(SEMANTIC_DIM) $^ $(LDLIBS) -o $@

semantic-search: $(BUILD_DIR)/semantic_search

$(BUILD_DIR)/benchmark_hnsw: benchmarks/benchmark_hnsw.c $(CORE_SRC) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DDIM=$(BENCH_DIM) $^ $(LDLIBS) -o $@

benchmark: $(BUILD_DIR)/benchmark_hnsw
	./$(BUILD_DIR)/benchmark_hnsw

clean:
	rm -rf $(BUILD_DIR)
