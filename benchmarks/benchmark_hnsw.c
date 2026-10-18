#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hnsw.h"
#include "bruteforce.h"

enum
{
    CLUSTER_COUNT = 8,
    QUERY_COUNT = 40,
    K = 5,
    EF_CONSTRUCTION = 100
};

static const int DATASET_SIZES[] = {250, 750, 1500};
static const int M_VALUES[] = {8, 16};
static const int EF_SEARCH_VALUES[] = {16, 64, 128};

/* -------------------- Deterministic data generation -------------------- */

/*
 * A small deterministic PRNG for generating reproducible synthetic data.
 * It is separate from rand(), which HNSW uses for random level assignment.
 */
static uint32_t next_random(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}


static float random_signed(uint32_t *state)
{
    double value =
        (double)next_random(state) / (double)UINT32_MAX;

    return (float)(2.0 * value - 1.0);
}


/*
 * Generate normalized cluster centres and a clustered synthetic dataset.
 *
 * Clustered data is more informative than a simple 1D line because
 * retrieval must navigate multiple regions of a multidimensional space.
 */
static void generate_dataset(
    Vector *vectors,
    int count,
    Vector centres[CLUSTER_COUNT]
)
{
    uint32_t state = UINT32_C(20261017);

    for (int c = 0; c < CLUSTER_COUNT; c++)
    {
        float norm_squared = 0.0f;

        for (int d = 0; d < DIM; d++)
        {
            centres[c].values[d] = random_signed(&state);

            norm_squared +=
                centres[c].values[d] * centres[c].values[d];
        }

        float norm = sqrtf(norm_squared);

        if (norm > 0.0f)
        {
            for (int d = 0; d < DIM; d++)
            {
                centres[c].values[d] /= norm;
            }
        }
    }

    for (int i = 0; i < count; i++)
    {
        int cluster = i % CLUSTER_COUNT;

        for (int d = 0; d < DIM; d++)
        {
            vectors[i].values[d] =
                centres[cluster].values[d]
                + 0.08f * random_signed(&state);
        }
    }
}


static void generate_queries(
    Vector queries[QUERY_COUNT],
    const Vector centres[CLUSTER_COUNT]
)
{
    uint32_t state = UINT32_C(314159265);

    for (int q = 0; q < QUERY_COUNT; q++)
    {
        int cluster = q % CLUSTER_COUNT;

        for (int d = 0; d < DIM; d++)
        {
            queries[q].values[d] =
                centres[cluster].values[d]
                + 0.08f * random_signed(&state);
        }
    }
}


/* -------------------- Timing and measurements -------------------- */

static double elapsed_nanoseconds(
    const struct timespec *start,
    const struct timespec *end
)
{
    return (double)(end->tv_sec - start->tv_sec) * 1.0e9
         + (double)(end->tv_nsec - start->tv_nsec);
}


/*
 * Count directed adjacency entries and estimate memory allocated
 * to the index's own structures.
 *
 * This excludes malloc allocator metadata, temporary search heaps,
 * dataset-generation buffers and benchmark query/result buffers.
 */
static void graph_statistics(
    const HNSWIndex *index,
    size_t *directed_edges,
    size_t *estimated_bytes
)
{
    size_t edges = 0;

    size_t bytes =
        sizeof(*index)
        + (size_t)index->capacity * sizeof(*index->nodes);

    for (int i = 0; i < index->count; i++)
    {
        const HNSWNode *node = index->nodes[i];

        bytes += sizeof(*node);

        bytes +=
            (size_t)(node->max_level + 1)
            * sizeof(*node->neighbors);

        for (int level = 0; level <= node->max_level; level++)
        {
            const NeighborList *list =
                &node->neighbors[level];

            edges += (size_t)list->count;

            bytes +=
                (size_t)list->capacity * sizeof(*list->ids);
        }
    }

    *directed_edges = edges;
    *estimated_bytes = bytes;
}


/*
 * Recall compares node IDs, not heap order.
 * The denominator is the number of exact neighbours requested.
 */
static float recall_at_k(
    const Heap *approximate,
    const Heap *exact
)
{
    if (approximate == NULL ||
        exact == NULL ||
        exact->size == 0)
    {
        return 0.0f;
    }

    int matches = 0;

    for (int i = 0; i < approximate->size; i++)
    {
        int candidate_id =
            approximate->items[i].node_id;

        int found = 0;

        for (int j = 0; j < exact->size; j++)
        {
            if (exact->items[j].node_id == candidate_id)
            {
                found = 1;
                break;
            }
        }

        if (found)
        {
            matches++;
        }
    }

    return (float)matches / (float)exact->size;
}


static void free_exact_results(
    Heap *results[QUERY_COUNT]
)
{
    for (int q = 0; q < QUERY_COUNT; q++)
    {
        heap_free(results[q]);
        results[q] = NULL;
    }
}


/* -------------------- One dataset / one M configuration -------------------- */

static int benchmark_configuration(
    const Vector *vectors,
    int dataset_size,
    const Vector queries[QUERY_COUNT],
    int M
)
{
    HNSWIndex *index = hnsw_create(M);

    if (index == NULL)
    {
        fprintf(stderr, "Failed to create index for M=%d.\n", M);
        return 0;
    }

    if (!hnsw_set_ef_construction(index, EF_CONSTRUCTION))
    {
        fprintf(stderr, "Failed to set ef_construction.\n");
        hnsw_destroy(index);
        return 0;
    }

    /*
     * Make HNSW's random level assignments reproducible
     * on the same C library and platform.
     */
    srand((unsigned int)(12345U + (unsigned int)dataset_size
                          + 31U * (unsigned int)M));

    struct timespec build_start;
    struct timespec build_end;

    if (clock_gettime(CLOCK_MONOTONIC, &build_start) != 0)
    {
        hnsw_destroy(index);
        return 0;
    }

    for (int i = 0; i < dataset_size; i++)
    {
        int id = hnsw_insert(index, vectors[i]);

        if (id != i)
        {
            fprintf(stderr,
                    "Insertion failed: N=%d, M=%d, vector=%d.\n",
                    dataset_size, M, i);

            hnsw_destroy(index);
            return 0;
        }
    }

    if (clock_gettime(CLOCK_MONOTONIC, &build_end) != 0)
    {
        hnsw_destroy(index);
        return 0;
    }

    double build_ms =
        elapsed_nanoseconds(&build_start, &build_end) / 1.0e6;

    size_t edge_count;
    size_t estimated_bytes;

    graph_statistics(index, &edge_count, &estimated_bytes);

    /*
     * Calculate exact answers for the same index and query set.
     * Reuse them for every efSearch value.
     */
    Heap *exact_results[QUERY_COUNT] = {NULL};
    double exact_total_ns = 0.0;

    for (int q = 0; q < QUERY_COUNT; q++)
    {
        struct timespec start;
        struct timespec end;

        if (clock_gettime(CLOCK_MONOTONIC, &start) != 0)
        {
            free_exact_results(exact_results);
            hnsw_destroy(index);
            return 0;
        }

        exact_results[q] =
            bruteforce_search(index, &queries[q], K);

        if (clock_gettime(CLOCK_MONOTONIC, &end) != 0)
        {
            free_exact_results(exact_results);
            hnsw_destroy(index);
            return 0;
        }

        exact_total_ns += elapsed_nanoseconds(&start, &end);

        if (exact_results[q] == NULL ||
            exact_results[q]->size != K)
        {
            fprintf(stderr, "Exact search failed for query %d.\n", q);

            free_exact_results(exact_results);
            hnsw_destroy(index);
            return 0;
        }
    }

    double exact_avg_ns =
        exact_total_ns / (double)QUERY_COUNT;

    double exact_avg_us = exact_avg_ns / 1000.0;

    double exact_qps =
        exact_avg_ns > 0.0 ? 1.0e9 / exact_avg_ns : 0.0;

    /*
     * Measure each query-breadth configuration on exactly
     * the same index and exact ground-truth results.
     */
    for (size_t e = 0;
         e < sizeof(EF_SEARCH_VALUES) / sizeof(EF_SEARCH_VALUES[0]);
         e++)
    {
        int ef_search = EF_SEARCH_VALUES[e];

        double hnsw_total_ns = 0.0;
        double recall_total = 0.0;

        for (int q = 0; q < QUERY_COUNT; q++)
        {
            struct timespec start;
            struct timespec end;

            if (clock_gettime(CLOCK_MONOTONIC, &start) != 0)
            {
                free_exact_results(exact_results);
                hnsw_destroy(index);
                return 0;
            }

            Heap *approximate =
                hnsw_search(index, &queries[q], K, ef_search);

            if (clock_gettime(CLOCK_MONOTONIC, &end) != 0)
            {
                heap_free(approximate);
                free_exact_results(exact_results);
                hnsw_destroy(index);
                return 0;
            }

            hnsw_total_ns += elapsed_nanoseconds(&start, &end);

            if (approximate == NULL)
            {
                fprintf(stderr,
                        "HNSW query failed: N=%d, M=%d, ef=%d, q=%d.\n",
                        dataset_size, M, ef_search, q);

                free_exact_results(exact_results);
                hnsw_destroy(index);
                return 0;
            }

            recall_total +=
                recall_at_k(approximate, exact_results[q]);

            heap_free(approximate);
        }

        double hnsw_avg_ns =
            hnsw_total_ns / (double)QUERY_COUNT;

        double hnsw_avg_us = hnsw_avg_ns / 1000.0;

        double hnsw_qps =
            hnsw_avg_ns > 0.0 ? 1.0e9 / hnsw_avg_ns : 0.0;

        double average_recall =
            recall_total / (double)QUERY_COUNT;

        /*
         * One CSV row per (N, M, efSearch) configuration.
         */
        printf(
            "%d,%d,%d,%d,%.3f,%.3f,%.3f,%.1f,%.1f,%.4f,%zu,%zu\n",
            dataset_size,
            M,
            EF_CONSTRUCTION,
            ef_search,
            build_ms,
            exact_avg_us,
            hnsw_avg_us,
            exact_qps,
            hnsw_qps,
            average_recall,
            edge_count,
            estimated_bytes
        );
    }

    free_exact_results(exact_results);
    hnsw_destroy(index);

    return 1;
}


/* -------------------- Program entry point -------------------- */

int main(void)
{
    const int dataset_count =
        (int)(sizeof(DATASET_SIZES) / sizeof(DATASET_SIZES[0]));

    /*
     * Machine-readable output: a CSV header followed by results.
     */
    printf(
        "dataset_size,M,ef_construction,ef_search,"
        "build_ms,brute_force_us_per_query,hnsw_us_per_query,"
        "brute_force_qps,hnsw_qps,recall_at_5,"
        "directed_edges,estimated_index_bytes\n"
    );

    for (int n = 0; n < dataset_count; n++)
    {
        int dataset_size = DATASET_SIZES[n];

        Vector *vectors =
            calloc((size_t)dataset_size, sizeof(*vectors));

        if (vectors == NULL)
        {
            fprintf(stderr, "Dataset allocation failed.\n");
            return EXIT_FAILURE;
        }

        Vector centres[CLUSTER_COUNT];
        Vector queries[QUERY_COUNT];

        generate_dataset(vectors, dataset_size, centres);
        generate_queries(queries, centres);

        for (size_t m = 0;
             m < sizeof(M_VALUES) / sizeof(M_VALUES[0]);
             m++)
        {
            if (!benchmark_configuration(
                    vectors,
                    dataset_size,
                    queries,
                    M_VALUES[m]))
            {
                free(vectors);
                return EXIT_FAILURE;
            }
        }

        free(vectors);
    }

    return EXIT_SUCCESS;
}
