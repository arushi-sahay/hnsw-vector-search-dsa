#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

#include "hnsw.h"

/*
 * Exact brute-force nearest-neighbor search.
 *
 * Examines every vector in the index.
 *
 * Returns a MAX_HEAP containing up to k nearest results.
 * The caller owns the returned heap and must call heap_free().
 *
 * Equal-distance ties are considered equally near.
 * Returns NULL for invalid input or search failure.
 */
Heap *bruteforce_search(
    const HNSWIndex *index,
    const Vector *query,
    int k
);

#endif
