#include "bruteforce.h"
#include <stddef.h>
#include <math.h>

/*
 * Exact top-k search by examining every stored vector.
 *
 * The result heap is a MAX_HEAP, so its root is the
 * worst currently retained result.
 */
Heap *bruteforce_search(
    const HNSWIndex *index,
    const Vector *query,
    int k
)
{
    if (index == NULL ||
        query == NULL ||
        index->nodes == NULL ||
        index->count <= 0 ||
        index->capacity < index->count ||
        k <= 0)
    {
        return NULL;
    }

    /*
     * There cannot be more distinct results than stored nodes.
     */
    int result_limit =
        (k < index->count) ? k : index->count;

    Heap *results = heap_create(result_limit, MAX_HEAP);

    if (results == NULL)
    {
        return NULL;
    }

    /*
     * Exact search: examine every vector in the index.
     */
    for (int i = 0; i < index->count; i++)
    {
        HNSWNode *node = index->nodes[i];

        if (node == NULL || node->id != i)
        {
            heap_free(results);
            return NULL;
        }

        float distance =
            euclidean_distance(query, &node->vector);

        if (!isfinite(distance))
        {
            heap_free(results);
            return NULL;
        }

        HeapItem candidate = {
            distance,
            node->id
        };

        /*
         * Until the heap contains k results, accept candidates.
         */
        if (results->size < result_limit)
        {
            if (!heap_push(results, candidate))
            {
                heap_free(results);
                return NULL;
            }

            continue;
        }

        /*
         * The root is the worst retained result.
         * Replace it only when this candidate is strictly closer.
         */
        HeapItem worst;

        if (!heap_peek(results, &worst))
        {
            heap_free(results);
            return NULL;
        }

        if (candidate.distance < worst.distance)
        {
            HeapItem removed;

            if (!heap_pop(results, &removed) ||
                !heap_push(results, candidate))
            {
                heap_free(results);
                return NULL;
            }
        }
    }

    return results;
}
