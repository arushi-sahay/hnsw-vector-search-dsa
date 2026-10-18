#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bruteforce.h"
#include "hnsw.h"

#define HNSW_M 8
#define EF_CONSTRUCTION 100
#define DEFAULT_K 5
#define DEFAULT_EF_SEARCH 32

typedef struct
{
    char **items;
    size_t count;
    size_t capacity;
} SentenceList;

/*
 * Read an entire line, including lines longer than a fixed buffer.
 *
 * Returns:
 *   1  a line was read
 *   0  end of file
 *  -1  allocation or file-reading error
 *
 * The caller owns the returned line.
 */
static int read_line(FILE *stream, char **line_out)
{
    if (stream == NULL || line_out == NULL)
    {
        return -1;
    }

    *line_out = NULL;

    size_t capacity = 128;
    size_t length = 0;
    char *line = malloc(capacity);

    if (line == NULL)
    {
        return -1;
    }

    int ch;

    while ((ch = fgetc(stream)) != EOF)
    {
        if (ch == '\n')
        {
            break;
        }

        if (length + 1 >= capacity)
        {
            if (capacity > SIZE_MAX / 2)
            {
                free(line);
                return -1;
            }

            size_t new_capacity = capacity * 2;
            char *grown = realloc(line, new_capacity);

            if (grown == NULL)
            {
                free(line);
                return -1;
            }

            line = grown;
            capacity = new_capacity;
        }

        line[length++] = (char)ch;
    }

    if (ch == EOF && ferror(stream))
    {
        free(line);
        return -1;
    }

    if (ch == EOF && length == 0)
    {
        free(line);
        return 0;
    }

    /* Handle Windows-style CRLF files too. */
    if (length > 0 && line[length - 1] == '\r')
    {
        length--;
    }

    line[length] = '\0';
    *line_out = line;

    return 1;
}

/*
 * Parse one headerless CSV row into a Vector.
 *
 * The row must contain exactly DIM finite floats and represent
 * an approximately unit-length vector. The latter requirement
 * matches the normalized sentence embeddings generated in Python.
 */
static int parse_vector_row(const char *line, Vector *vector)
{
    if (line == NULL || vector == NULL)
    {
        return 0;
    }

    const char *cursor = line;
    double norm_squared = 0.0;

    for (int dimension = 0; dimension < DIM; dimension++)
    {
        char *end = NULL;

        errno = 0;
        float value = strtof(cursor, &end);

        if (end == cursor || errno == ERANGE || !isfinite(value))
        {
            return 0;
        }

        vector->values[dimension] = value;
        norm_squared += (double)value * (double)value;

        if (dimension < DIM - 1)
        {
            if (*end != ',')
            {
                return 0;
            }

            cursor = end + 1;
        }
        else if (*end != '\0')
        {
            return 0;
        }
    }

    double norm = sqrt(norm_squared);

    if (!isfinite(norm) || fabs(norm - 1.0) > 0.001)
    {
        return 0;
    }

    return 1;
}

static int append_sentence(SentenceList *sentences, char *sentence)
{
    if (sentences == NULL || sentence == NULL)
    {
        return 0;
    }

    if (sentences->count == sentences->capacity)
    {
        size_t new_capacity =
            (sentences->capacity == 0) ? 16 : sentences->capacity * 2;

        if (new_capacity < sentences->capacity ||
            new_capacity > SIZE_MAX / sizeof(*sentences->items))
        {
            return 0;
        }

        char **grown = realloc(
            sentences->items,
            new_capacity * sizeof(*sentences->items)
        );

        if (grown == NULL)
        {
            return 0;
        }

        sentences->items = grown;
        sentences->capacity = new_capacity;
    }

    /* Ownership transfers to SentenceList on success. */
    sentences->items[sentences->count++] = sentence;

    return 1;
}

static void free_sentences(SentenceList *sentences)
{
    if (sentences == NULL)
    {
        return;
    }

    for (size_t i = 0; i < sentences->count; i++)
    {
        free(sentences->items[i]);
    }

    free(sentences->items);

    sentences->items = NULL;
    sentences->count = 0;
    sentences->capacity = 0;
}

/*
 * Load aligned sentence/vector files and insert vectors into HNSW.
 *
 * Critical invariant:
 *   vector row i <-> sentence line i <-> HNSW node ID i
 *
 * We reject mismatched files rather than silently corrupting this
 * mapping.
 */
static int load_corpus(
    HNSWIndex *index,
    const char *vectors_path,
    const char *sentences_path,
    SentenceList *sentences
)
{
    FILE *vectors_file = fopen(vectors_path, "r");

    if (vectors_file == NULL)
    {
        perror(vectors_path);
        return 0;
    }

    FILE *sentences_file = fopen(sentences_path, "r");

    if (sentences_file == NULL)
    {
        perror(sentences_path);
        fclose(vectors_file);
        return 0;
    }

    int success = 0;
    size_t row_number = 0;

    for (;;)
    {
        char *vector_line = NULL;
        char *sentence_line = NULL;

        int vector_status = read_line(vectors_file, &vector_line);
        int sentence_status = read_line(sentences_file, &sentence_line);

        if (vector_status < 0 || sentence_status < 0)
        {
            fprintf(stderr, "Error reading corpus files.\n");
            free(vector_line);
            free(sentence_line);
            break;
        }

        if (vector_status == 0 && sentence_status == 0)
        {
            success = 1;
            break;
        }

        if (vector_status != 1 || sentence_status != 1)
        {
            fprintf(
                stderr,
                "Corpus files have different line counts near row %zu.\n",
                row_number + 1
            );

            free(vector_line);
            free(sentence_line);
            break;
        }

        if (sentence_line[0] == '\0')
        {
            fprintf(
                stderr,
                "Empty sentence at line %zu; refusing to shift node IDs.\n",
                row_number + 1
            );

            free(vector_line);
            free(sentence_line);
            break;
        }

        Vector vector;

        if (!parse_vector_row(vector_line, &vector))
        {
            fprintf(
                stderr,
                "Invalid vector at row %zu. Expected exactly %d finite "
                "comma-separated values and an approximately unit L2 norm.\n",
                row_number + 1,
                DIM
            );

            free(vector_line);
            free(sentence_line);
            break;
        }

        if (sentences->count >= (size_t)INT_MAX)
        {
            fprintf(stderr, "Corpus is too large for integer node IDs.\n");
            free(vector_line);
            free(sentence_line);
            break;
        }

        int expected_id = (int)sentences->count;
        int inserted_id = hnsw_insert(index, vector);

        if (inserted_id != expected_id)
        {
            fprintf(
                stderr,
                "HNSW insertion failed for row %zu (returned ID %d).\n",
                row_number + 1,
                inserted_id
            );

            free(vector_line);
            free(sentence_line);
            break;
        }

        if (!append_sentence(sentences, sentence_line))
        {
            fprintf(stderr, "Could not allocate sentence storage.\n");
            free(vector_line);
            free(sentence_line);
            break;
        }

        /* The sentence list now owns sentence_line. */
        free(vector_line);
        row_number++;
    }

    if (fclose(vectors_file) != 0)
    {
        perror(vectors_path);
        success = 0;
    }

    if (fclose(sentences_file) != 0)
    {
        perror(sentences_path);
        success = 0;
    }

    if (success && sentences->count == 0)
    {
        fprintf(stderr, "The corpus contains no vectors.\n");
        success = 0;
    }

    if (success && index->count != (int)sentences->count)
    {
        fprintf(stderr, "Internal error: index and corpus counts differ.\n");
        success = 0;
    }

    if (success)
    {
        printf("Loaded %zu sentences into the HNSW index.\n",
               sentences->count);
    }

    return success;
}

static int load_query_vector(
    const char *path,
    Vector *query
)
{
    FILE *file = fopen(path, "r");

    if (file == NULL)
    {
        perror(path);
        return 0;
    }

    char *line = NULL;
    int status = read_line(file, &line);

    if (status != 1 || line == NULL || line[0] == '\0')
    {
        fprintf(stderr, "Query file must contain exactly one vector row.\n");
        free(line);
        fclose(file);
        return 0;
    }

    if (!parse_vector_row(line, query))
    {
        fprintf(
            stderr,
            "Invalid query vector. Expected %d finite values and an "
            "approximately unit L2 norm.\n",
            DIM
        );

        free(line);
        fclose(file);
        return 0;
    }

    free(line);

    char *extra_line = NULL;
    status = read_line(file, &extra_line);

    if (status != 0)
    {
        fprintf(stderr, "Query file must contain exactly one vector row.\n");
        free(extra_line);
        fclose(file);
        return 0;
    }

    if (fclose(file) != 0)
    {
        perror(path);
        return 0;
    }

    return 1;
}

static int parse_positive_int(const char *text, int *value_out)
{
    if (text == NULL || value_out == NULL || *text == '\0')
    {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);

    if (errno != 0 ||
        end == text ||
        *end != '\0' ||
        value < 1 ||
        value > INT_MAX)
    {
        return 0;
    }

    *value_out = (int)value;
    return 1;
}

static int compare_heap_items(const void *left, const void *right)
{
    const HeapItem *a = left;
    const HeapItem *b = right;

    if (a->distance < b->distance)
    {
        return -1;
    }

    if (a->distance > b->distance)
    {
        return 1;
    }

    if (a->node_id < b->node_id)
    {
        return -1;
    }

    if (a->node_id > b->node_id)
    {
        return 1;
    }

    return 0;
}

/*
 * Drain a result heap into an array ordered nearest-first.
 *
 * The heap API returns results from a max-heap, so its internal array
 * must not be assumed to already be sorted.
 */
static int heap_to_sorted_array(
    Heap *heap,
    HeapItem **items_out,
    size_t *count_out
)
{
    if (heap == NULL || items_out == NULL || count_out == NULL ||
        heap->size < 0)
    {
        return 0;
    }

    *items_out = NULL;
    *count_out = 0;

    size_t expected_count = (size_t)heap->size;

    if (expected_count > SIZE_MAX / sizeof(HeapItem))
    {
        return 0;
    }

    HeapItem *items = NULL;

    if (expected_count > 0)
    {
        items = malloc(expected_count * sizeof(*items));

        if (items == NULL)
        {
            return 0;
        }
    }

    size_t count = 0;

    while (!heap_is_empty(heap))
    {
        HeapItem item;

        if (!heap_pop(heap, &item) || count >= expected_count)
        {
            free(items);
            return 0;
        }

        items[count++] = item;
    }

    if (count != expected_count)
    {
        free(items);
        return 0;
    }

    if (count > 1)
    {
        qsort(items, count, sizeof(*items), compare_heap_items);
    }

    *items_out = items;
    *count_out = count;

    return 1;
}

/*
 * For unit-length vectors:
 *
 *   squared_Euclidean_distance = 2 - 2*cosine_similarity
 *
 * Therefore cosine similarity = 1 - distance^2 / 2.
 */
static double cosine_similarity_from_distance(float distance)
{
    double d = (double)distance;
    double similarity = 1.0 - (d * d / 2.0);

    /* Protect the display against tiny floating-point overshoots. */
    if (similarity > 1.0)
    {
        similarity = 1.0;
    }

    if (similarity < -1.0)
    {
        similarity = -1.0;
    }

    return similarity;
}

static void print_exact_ids(
    const HeapItem *items,
    size_t count
)
{
    printf("\nExact brute-force results (ID and Euclidean distance):\n");

    for (size_t i = 0; i < count; i++)
    {
        printf(
            "  %zu. ID=%d, distance=%.6f\n",
            i + 1,
            items[i].node_id,
            items[i].distance
        );
    }
}

static void print_usage(const char *program)
{
    fprintf(
        stderr,
        "Usage: %s <embeddings.csv> <sentences.txt> "
        "<query_embedding.csv> [k] [ef_search]\n"
        "\n"
        "Defaults: k=%d, ef_search=%d\n",
        program,
        DEFAULT_K,
        DEFAULT_EF_SEARCH
    );
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 6)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int k = DEFAULT_K;
    int ef_search = DEFAULT_EF_SEARCH;

    if (argc >= 5 && !parse_positive_int(argv[4], &k))
    {
        fprintf(stderr, "k must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    if (argc >= 6 && !parse_positive_int(argv[5], &ef_search))
    {
        fprintf(stderr, "ef_search must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    HNSWIndex *index = NULL;
    Heap *approximate_heap = NULL;
    Heap *exact_heap = NULL;

    HeapItem *approximate_items = NULL;
    HeapItem *exact_items = NULL;

    size_t approximate_count = 0;
    size_t exact_count = 0;

    SentenceList sentences = {0};
    Vector query;

    int exit_status = EXIT_FAILURE;

    /* Make the graph's random-level assignment repeatable. */
    srand(42U);

    index = hnsw_create(HNSW_M);

    if (index == NULL)
    {
        fprintf(stderr, "Could not create HNSW index.\n");
        goto cleanup;
    }

    if (!hnsw_set_ef_construction(index, EF_CONSTRUCTION))
    {
        fprintf(stderr, "Could not configure ef_construction.\n");
        goto cleanup;
    }

    if (!load_corpus(index, argv[1], argv[2], &sentences))
    {
        goto cleanup;
    }

    if (!load_query_vector(argv[3], &query))
    {
        goto cleanup;
    }

    approximate_heap = hnsw_search(index, &query, k, ef_search);

    if (approximate_heap == NULL)
    {
        fprintf(stderr, "HNSW query failed.\n");
        goto cleanup;
    }

    exact_heap = bruteforce_search(index, &query, k);

    if (exact_heap == NULL)
    {
        fprintf(stderr, "Brute-force query failed.\n");
        goto cleanup;
    }

    if (!heap_to_sorted_array(
            approximate_heap,
            &approximate_items,
            &approximate_count) ||
        !heap_to_sorted_array(
            exact_heap,
            &exact_items,
            &exact_count))
    {
        fprintf(stderr, "Could not extract search results.\n");
        goto cleanup;
    }

    if (approximate_count == 0 ||
        exact_count == 0 ||
        approximate_count != exact_count)
    {
        fprintf(stderr, "Unexpected number of search results.\n");
        goto cleanup;
    }

    printf("\n=== Semantic Search Results ===\n");
    printf(
        "Vectors: %d | Dimensions: %d | M: %d | "
        "ef_construction: %d | ef_search: %d | requested k: %d\n",
        index->count,
        DIM,
        index->M,
        index->ef_construction,
        ef_search,
        k
    );

    printf(
        "\nApproximate HNSW results "
        "(ordered nearest-first):\n"
    );

    for (size_t i = 0; i < approximate_count; i++)
    {
        int node_id = approximate_items[i].node_id;

        if (node_id < 0 || (size_t)node_id >= sentences.count)
        {
            fprintf(stderr, "Search returned an invalid node ID.\n");
            goto cleanup;
        }

        float distance = approximate_items[i].distance;
        double similarity = cosine_similarity_from_distance(distance);

        printf(
            "\n%zu. ID=%d | Euclidean distance=%.6f | "
            "cosine similarity=%.6f\n",
            i + 1,
            node_id,
            distance,
            similarity
        );

        printf("   %s\n", sentences.items[(size_t)node_id]);
    }

    print_exact_ids(exact_items, exact_count);

    /*
     * Recall@k = number of approximate results also found in the
     * exact top-k / number of exact top-k results.
     *
     * Distinct IDs are compared; both search routines return unique
     * nodes in the existing implementation.
     */
    size_t overlap = 0;

    for (size_t i = 0; i < approximate_count; i++)
    {
        for (size_t j = 0; j < exact_count; j++)
        {
            if (approximate_items[i].node_id == exact_items[j].node_id)
            {
                overlap++;
                break;
            }
        }
    }

    double recall = (double)overlap / (double)exact_count;

    printf(
        "\nRecall@%zu: %zu/%zu = %.2f%%\n",
        exact_count,
        overlap,
        exact_count,
        recall * 100.0
    );

    exit_status = EXIT_SUCCESS;

cleanup:
    free(approximate_items);
    free(exact_items);

    heap_free(approximate_heap);
    heap_free(exact_heap);

    free_sentences(&sentences);
    hnsw_destroy(index);

    return exit_status;
}
