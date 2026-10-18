#ifndef HNSW_H
#define HNSW_H

#include "node.h"
#include "heap.h"

typedef struct
{
    HNSWNode **nodes;
    int count;
    int capacity;

    int entry_point;
    int max_level;

    int M;
    int ef_construction;
} HNSWIndex;

/* Index lifecycle and configuration. */
HNSWIndex *hnsw_create(int M);
void hnsw_destroy(HNSWIndex *index);
int hnsw_set_ef_construction(HNSWIndex *index, int ef);

/* Node storage and insertion. */
int hnsw_add_node(HNSWIndex *index, HNSWNode *node);
int hnsw_insert(HNSWIndex *index, Vector vector);
int random_level(const HNSWIndex *index);

/* Search operations. */
int greedy_search_layer(
    const HNSWIndex *index,
    const Vector *query,
    int entry_id,
    int level
);

Heap *hnsw_search_layer(
    const HNSWIndex *index,
    const Vector *query,
    int entry_id,
    int level,
    int ef
);

/* Neighbor selection and connection management. */
int hnsw_select_neighbors(
    const HNSWIndex *index,
    const Vector *query,
    const Heap *candidates,
    int max_neighbors,
    NeighborList *selected
);

int hnsw_prune_neighbors(
    HNSWIndex *index,
    int node_id,
    int level
);

int hnsw_connect_nodes(
    HNSWIndex *index,
    int a_id,
    int b_id,
    int level
);

/*
 * Search the index for the k nearest vectors.
 *
 * ef_search controls the search breadth. The effective
 * value is at least k.
 *
 * Returns a max-heap containing up to k nearest results.
 * The caller must release it with heap_free().
 *
 * Returns NULL for invalid input, an empty index,
 * or an allocation/search failure.
 */
Heap *hnsw_search(
    const HNSWIndex *index,
    const Vector *query,
    int k,
    int ef_search
);

#endif
