#include "hnsw.h"
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>

HNSWIndex *hnsw_create(int M)
{
    if (M < 2 || M > INT_MAX / 2)
    {
        return NULL;
    }

    HNSWIndex *index = malloc(sizeof(*index));

    if (index == NULL)
    {
        return NULL;
    }

    index->nodes = NULL;
    index->count = 0;
    index->capacity = 0;

    index->entry_point = -1;
    index->max_level = -1;

    index->M = M;
    index->ef_construction = 100;

    return index;
}

int random_level(const HNSWIndex *index)
{
    double u = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);

    double level_multiplier = 1.0 / log((double) index->M);

    int level = (int) (-log(u) * level_multiplier);

    return level;
}

int hnsw_add_node(HNSWIndex *index, HNSWNode *node)
{
    if (index == NULL ||
        node == NULL ||
        index->count < 0 ||
        index->capacity < 0 ||
        index->count > index->capacity ||
        (index->capacity > 0 && index->nodes == NULL) ||
        node->id != index->count ||
        node->max_level < 0 ||
        node->neighbors == NULL)
    {
        return 0;
    }

    if (index->count == INT_MAX)
    {
        return 0;
    }

    if (index->count == index->capacity)
    {
        int new_capacity;

        if (index->capacity == 0)
        {
            new_capacity = 4;
        }
        else
        {
            if (index->capacity > INT_MAX / 2)
            {
                return 0;
            }

            new_capacity = index->capacity * 2;
        }

        if ((size_t)new_capacity > SIZE_MAX / sizeof(*index->nodes))
        {
            return 0;
        }

        HNSWNode **new_nodes = realloc(
            index->nodes,
            (size_t)new_capacity * sizeof(*index->nodes)
        );

        if (new_nodes == NULL)
        {
            return 0;
        }

        index->nodes = new_nodes;
        index->capacity = new_capacity;
    }

    int was_empty = (index->count == 0);

    index->nodes[index->count] = node;
    index->count++;

    /*
     * Keep the entry point at a node with the highest
     * maximum level currently present in the index.
     */
    if (was_empty || node->max_level > index->max_level)
    {
        index->entry_point = node->id;
        index->max_level = node->max_level;
    }

    return 1;
}

int greedy_search_layer(
    const HNSWIndex *index,
    const Vector *query,
    int entry_id,
    int level
)
{
    if (index == NULL ||
        query == NULL ||
        entry_id < 0 ||
        entry_id >= index->count ||
        level < 0)
    {
        return -1;
    }

    HNSWNode *current = index->nodes[entry_id];

    if (current == NULL || level > current->max_level)
    {
        return -1;
    }

    float current_distance =
        euclidean_distance(query, &current->vector);

    while (1)
    {
        int best_id = current->id;
        float best_distance = current_distance;

        NeighborList *neighbors =
            &current->neighbors[level];

        for (int i = 0; i < neighbors->count; i++)
        {
            int neighbor_id = neighbors->ids[i];

            if (neighbor_id < 0 ||
                neighbor_id >= index->count)
            {
                continue;
            }

            HNSWNode *neighbor = index->nodes[neighbor_id];

            if (neighbor == NULL ||
                level > neighbor->max_level)
            {
                continue;
            }

            float neighbor_distance =
                euclidean_distance(query, &neighbor->vector);

            if (neighbor_distance < best_distance)
            {
                best_id = neighbor_id;
                best_distance = neighbor_distance;
            }
        }

        if (best_id == current->id)
        {
            break;
        }

        current = index->nodes[best_id];
        current_distance = best_distance;
    }

    return current->id;
}


Heap *hnsw_search_layer(
    const HNSWIndex *index,
    const Vector *query,
    int entry_id,
    int level,
    int ef
)
{
    if (index == NULL ||
        query == NULL ||
        index->nodes == NULL ||
        index->count <= 0 ||
        entry_id < 0 ||
        entry_id >= index->count ||
        level < 0 ||
        ef <= 0)
    {
        return NULL;
    }

    HNSWNode *entry = index->nodes[entry_id];

    if (entry == NULL ||
        entry->id != entry_id ||
        entry->max_level < level ||
        entry->neighbors == NULL)
    {
        return NULL;
    }

    int limit = (ef < index->count) ? ef : index->count;

    Heap *candidates = heap_create(limit, MIN_HEAP);
    Heap *results = heap_create(limit, MAX_HEAP);

    unsigned char *visited =
        calloc((size_t)index->count, sizeof(*visited));

    if (candidates == NULL ||
        results == NULL ||
        visited == NULL)
    {
        heap_free(candidates);
        heap_free(results);
        free(visited);
        return NULL;
    }

    float entry_distance =
        euclidean_distance(query, &entry->vector);

    HeapItem entry_item = {
        entry_distance,
        entry_id
    };

    if (!heap_push(candidates, entry_item) ||
        !heap_push(results, entry_item))
    {
        goto failure;
    }

    visited[entry_id] = 1;

    while (!heap_is_empty(candidates))
    {
        HeapItem current_item;

        if (!heap_pop(candidates, &current_item))
        {
            goto failure;
        }

        /*
         * If the nearest remaining candidate is already
         * worse than our worst retained result, stop.
         */
        if (results->size >= limit)
        {
            HeapItem worst;

            if (!heap_peek(results, &worst))
            {
                goto failure;
            }

            if (current_item.distance > worst.distance)
            {
                break;
            }
        }

        int current_id = current_item.node_id;
        HNSWNode *current = index->nodes[current_id];

        if (current == NULL ||
            current->id != current_id ||
            current->max_level < level ||
            current->neighbors == NULL)
        {
            continue;
        }

        NeighborList *neighbors =
            &current->neighbors[level];

        if (neighbors->count < 0 ||
            neighbors->count > neighbors->capacity ||
            (neighbors->count > 0 && neighbors->ids == NULL))
        {
            goto failure;
        }

        for (int i = 0; i < neighbors->count; i++)
        {
            int neighbor_id = neighbors->ids[i];

            if (neighbor_id < 0 ||
                neighbor_id >= index->count)
            {
                continue;
            }

            if (visited[neighbor_id])
            {
                continue;
            }

            HNSWNode *neighbor = index->nodes[neighbor_id];

            if (neighbor == NULL ||
                neighbor->id != neighbor_id ||
                neighbor->max_level < level ||
                neighbor->neighbors == NULL)
            {
                continue;
            }

            /*
             * Mark this node so it is not evaluated again.
             */
            visited[neighbor_id] = 1;

            float distance =
                euclidean_distance(query, &neighbor->vector);

            HeapItem item = {
                distance,
                neighbor_id
            };

            int should_add = (results->size < limit);

            if (!should_add)
            {
                HeapItem worst;

                if (!heap_peek(results, &worst))
                {
                    goto failure;
                }

                should_add = (distance < worst.distance);
            }

            if (!should_add)
            {
                continue;
            }

            /*
             * Add promising nodes to the exploration frontier.
             */
            if (!heap_push(candidates, item))
            {
                goto failure;
            }

            /*
             * Maintain at most 'limit' retained results.
             * When full, remove the worst before inserting
             * the new candidate.
             */
            if (results->size < limit)
            {
                if (!heap_push(results, item))
                {
                    goto failure;
                }
            }
            else
            {
                HeapItem removed;

                if (!heap_pop(results, &removed) ||
                    !heap_push(results, item))
                {
                    goto failure;
                }
            }
        }
    }

    heap_free(candidates);
    free(visited);

    /*
     * The caller owns this heap and must eventually
     * release it with heap_free().
     */
    return results;

failure:
    heap_free(candidates);
    heap_free(results);
    free(visited);

    return NULL;
}


/*
 * Check whether a neighbor ID is already present in a list.
 */
static int neighbor_list_contains_id(
    const NeighborList *list,
    int node_id
)
{
    for (int i = 0; i < list->count; i++)
    {
        if (list->ids[i] == node_id)
        {
            return 1;
        }
    }

    return 0;
}


/*
 * Sort candidates by increasing distance.
 * Break ties using node ID for deterministic ordering.
 */
static int compare_candidates(const void *a, const void *b)
{
    const HeapItem *item_a = a;
    const HeapItem *item_b = b;

    if (item_a->distance < item_b->distance)
    {
        return -1;
    }

    if (item_a->distance > item_b->distance)
    {
        return 1;
    }

    if (item_a->node_id < item_b->node_id)
    {
        return -1;
    }

    if (item_a->node_id > item_b->node_id)
    {
        return 1;
    }

    return 0;
}


/*
 * Select diverse neighbors from a candidate heap.
 *
 * The output list must initially be empty:
 *     NeighborList selected = {NULL, 0, 0};
 *
 * Candidates are considered in increasing distance from query.
 * A candidate is accepted if it is not closer to an already
 * selected neighbor than it is to the query.
 *
 * If fewer than max_neighbors are selected, the closest
 * rejected candidates are added as a fallback.
 *
 * Returns 1 on success and 0 on invalid input or allocation failure.
 */
int hnsw_select_neighbors(
    const HNSWIndex *index,
    const Vector *query,
    const Heap *candidates,
    int max_neighbors,
    NeighborList *selected
)
{
    if (index == NULL ||
        query == NULL ||
        candidates == NULL ||
        selected == NULL ||
        index->nodes == NULL ||
        index->count <= 0 ||
        max_neighbors <= 0 ||
        candidates->size < 0 ||
        candidates->capacity < 0 ||
        candidates->size > candidates->capacity ||
        (candidates->size > 0 && candidates->items == NULL) ||
        (candidates->type != MIN_HEAP &&
         candidates->type != MAX_HEAP) ||
        selected->ids != NULL ||
        selected->count != 0 ||
        selected->capacity != 0)
    {
        return 0;
    }

    int limit = max_neighbors;

    if (limit > index->count)
    {
        limit = index->count;
    }

    HeapItem *ordered = NULL;
    unsigned char *seen = NULL;

    NeighborList chosen = {NULL, 0, 0};

    if ((size_t)candidates->size > SIZE_MAX / sizeof(*ordered))
    {
        return 0;
    }

    if (candidates->size > 0)
    {
        ordered = malloc(
            (size_t)candidates->size * sizeof(*ordered)
        );

        if (ordered == NULL)
        {
            return 0;
        }
    }

    seen = calloc((size_t)index->count, sizeof(*seen));

    if (seen == NULL)
    {
        free(ordered);
        return 0;
    }

    /*
     * Copy valid candidates, recompute their distances,
     * and remove duplicate or invalid node IDs.
     *
     * The original heap remains unchanged.
     */
    int valid_count = 0;

    for (int i = 0; i < candidates->size; i++)
    {
        int node_id = candidates->items[i].node_id;

        if (node_id < 0 || node_id >= index->count)
        {
            continue;
        }

        if (seen[node_id])
        {
            continue;
        }

        seen[node_id] = 1;

        HNSWNode *node = index->nodes[node_id];

        if (node == NULL || node->id != node_id)
        {
            continue;
        }

        float distance =
            euclidean_distance(query, &node->vector);

        if (!isfinite(distance))
        {
            continue;
        }

        ordered[valid_count].distance = distance;
        ordered[valid_count].node_id = node_id;
        valid_count++;
    }

    /*
     * Heap array order is not globally sorted.
     * Sort our copy so candidates are considered nearest-first.
     */
    if (valid_count > 1)
    {
        qsort(
            ordered,
            (size_t)valid_count,
            sizeof(*ordered),
            compare_candidates
        );
    }

    /*
     * Phase 1: select diverse candidates.
     */
    for (int i = 0;
         i < valid_count && chosen.count < limit;
         i++)
    {
        int candidate_id = ordered[i].node_id;

        HNSWNode *candidate = index->nodes[candidate_id];

        int is_diverse = 1;

        for (int j = 0; j < chosen.count; j++)
        {
            int selected_id = chosen.ids[j];

            HNSWNode *selected_node = index->nodes[selected_id];

            float distance_to_selected =
                euclidean_distance(
                    &candidate->vector,
                    &selected_node->vector
                );

            /*
             * If the candidate is closer to an already
             * selected neighbor than to the query, it may
             * provide a redundant connection.
             */
            if (distance_to_selected < ordered[i].distance)
            {
                is_diverse = 0;
                break;
            }
        }

        if (is_diverse)
        {
            if (!neighbor_list_add(&chosen, candidate_id))
            {
                goto failure;
            }
        }
    }

    /*
     * Phase 2: fallback.
     *
     * If the diversity rule left unused slots, fill them
     * with the closest remaining candidates.
     */
    for (int i = 0;
         i < valid_count && chosen.count < limit;
         i++)
    {
        int candidate_id = ordered[i].node_id;

        if (neighbor_list_contains_id(&chosen, candidate_id))
        {
            continue;
        }

        if (!neighbor_list_add(&chosen, candidate_id))
        {
            goto failure;
        }
    }

    /*
     * Transfer ownership of the selected array to the caller.
     */
    *selected = chosen;

    free(ordered);
    free(seen);

    return 1;

failure:
    free(chosen.ids);
    free(ordered);
    free(seen);

    return 0;
}


/*
 * Validate a neighbor list before operating on it.
 */
static int connection_list_is_valid(const NeighborList *list)
{
    if (list == NULL ||
        list->count < 0 ||
        list->capacity < 0 ||
        list->count > list->capacity)
    {
        return 0;
    }

    if ((list->count > 0 || list->capacity > 0) &&
        list->ids == NULL)
    {
        return 0;
    }

    return 1;
}


/*
 * Remove one occurrence of a neighbor ID.
 * This removes an edge, not the node itself.
 */
static int neighbor_list_remove_id(
    NeighborList *list,
    int node_id
)
{
    if (!connection_list_is_valid(list))
    {
        return 0;
    }

    for (int i = 0; i < list->count; i++)
    {
        if (list->ids[i] == node_id)
        {
            for (int j = i + 1; j < list->count; j++)
            {
                list->ids[j - 1] = list->ids[j];
            }

            list->count--;
            return 1;
        }
    }

    return 0;
}


/*
 * Determine the maximum permitted degree for a layer.
 *
 * Layer 0: 2 * M
 * Upper layers: M
 */
static int connection_limit(const HNSWIndex *index, int level)
{
    if (index == NULL ||
        index->M < 2 ||
        index->M > INT_MAX / 2 ||
        level < 0)
    {
        return -1;
    }

    if (level == 0)
    {
        return 2 * index->M;
    }

    return index->M;
}


/*
 * Prune a node's neighbor list if it exceeds the limit.
 *
 * Neighbor selection uses the existing HNSW diversity
 * heuristic. Any removed edge is also removed from the
 * other endpoint, preserving bidirectional connectivity.
 *
 * Returns 1 on success, 0 on invalid input or failure.
 */
int hnsw_prune_neighbors(
    HNSWIndex *index,
    int node_id,
    int level
)
{
    if (index == NULL ||
        index->nodes == NULL ||
        index->count <= 0 ||
        node_id < 0 ||
        node_id >= index->count)
    {
        return 0;
    }

    int limit = connection_limit(index, level);

    if (limit < 0)
    {
        return 0;
    }

    HNSWNode *node = index->nodes[node_id];

    if (node == NULL ||
        node->id != node_id ||
        node->max_level < level ||
        node->neighbors == NULL)
    {
        return 0;
    }

    NeighborList *list = &node->neighbors[level];

    if (!connection_list_is_valid(list))
    {
        return 0;
    }

    /* Nothing needs pruning if the list is within its limit. */
    if (list->count <= limit)
    {
        return 1;
    }

    /*
     * Build candidates from this node's existing neighbors.
     * Distances are measured from the node being pruned.
     */
    Heap *candidates = heap_create(list->count, MIN_HEAP);

    if (candidates == NULL)
    {
        return 0;
    }

    for (int i = 0; i < list->count; i++)
    {
        int neighbor_id = list->ids[i];

        if (neighbor_id < 0 ||
            neighbor_id >= index->count ||
            neighbor_id == node_id)
        {
            continue;
        }

        HNSWNode *neighbor = index->nodes[neighbor_id];

        if (neighbor == NULL ||
            neighbor->id != neighbor_id ||
            neighbor->max_level < level ||
            neighbor->neighbors == NULL)
        {
            continue;
        }

        float distance =
            euclidean_distance(&node->vector, &neighbor->vector);

        HeapItem item = {distance, neighbor_id};

        if (!heap_push(candidates, item))
        {
            heap_free(candidates);
            return 0;
        }
    }

    NeighborList selected = {NULL, 0, 0};

    if (!hnsw_select_neighbors(
            index,
            &node->vector,
            candidates,
            limit,
            &selected))
    {
        heap_free(candidates);
        return 0;
    }

    heap_free(candidates);

    /*
     * Remove reverse edges for neighbors that were not retained.
     * The original list stays intact until selection succeeds.
     */
    for (int i = 0; i < list->count; i++)
    {
        int old_neighbor_id = list->ids[i];

        if (old_neighbor_id < 0 ||
            old_neighbor_id >= index->count ||
            old_neighbor_id == node_id)
        {
            continue;
        }

        if (neighbor_list_contains_id(&selected, old_neighbor_id))
        {
            continue;
        }

        HNSWNode *old_neighbor = index->nodes[old_neighbor_id];

        if (old_neighbor == NULL ||
            old_neighbor->id != old_neighbor_id ||
            old_neighbor->max_level < level ||
            old_neighbor->neighbors == NULL)
        {
            continue;
        }

        NeighborList *reverse =
            &old_neighbor->neighbors[level];

        if (connection_list_is_valid(reverse))
        {
            neighbor_list_remove_id(reverse, node_id);
        }
    }

    /*
     * Replace the old adjacency list with the selected one.
     * Ownership of selected.ids transfers to the node.
     */
    free(list->ids);
    *list = selected;

    return 1;
}


/*
 * Add a bidirectional edge between two nodes at a layer.
 *
 * If either endpoint exceeds its connection limit, its
 * neighbor list is pruned using the selection heuristic.
 *
 * Note: pruning may discard the newly added edge if it
 * is not among the selected connections.
 */
int hnsw_connect_nodes(
    HNSWIndex *index,
    int a_id,
    int b_id,
    int level
)
{
    if (index == NULL ||
        index->nodes == NULL ||
        index->count <= 0 ||
        a_id < 0 ||
        a_id >= index->count ||
        b_id < 0 ||
        b_id >= index->count ||
        a_id == b_id)
    {
        return 0;
    }

    int limit = connection_limit(index, level);

    if (limit < 0)
    {
        return 0;
    }

    HNSWNode *a = index->nodes[a_id];
    HNSWNode *b = index->nodes[b_id];

    if (a == NULL ||
        b == NULL ||
        a->id != a_id ||
        b->id != b_id ||
        a->max_level < level ||
        b->max_level < level ||
        a->neighbors == NULL ||
        b->neighbors == NULL)
    {
        return 0;
    }

    NeighborList *list_a = &a->neighbors[level];
    NeighborList *list_b = &b->neighbors[level];

    if (!connection_list_is_valid(list_a) ||
        !connection_list_is_valid(list_b))
    {
        return 0;
    }

    int a_has_b = neighbor_list_contains_id(list_a, b_id);
    int b_has_a = neighbor_list_contains_id(list_b, a_id);

    int added_to_a = 0;

    /*
     * Avoid duplicate edges, but repair a missing reverse
     * edge if the existing graph is asymmetric.
     */
    if (!a_has_b)
    {
        if (!neighbor_list_add(list_a, b_id))
        {
            return 0;
        }

        added_to_a = 1;
    }

    if (!b_has_a)
    {
        if (!neighbor_list_add(list_b, a_id))
        {
            if (added_to_a)
            {
                neighbor_list_remove_id(list_a, b_id);
            }

            return 0;
        }
    }

    /*
     * Prune both endpoints. Pruning one endpoint may remove
     * an edge from the other endpoint's adjacency list too.
     */
    if (!hnsw_prune_neighbors(index, a_id, level))
    {
        return 0;
    }

    if (!hnsw_prune_neighbors(index, b_id, level))
    {
        return 0;
    }

    return 1;
}


/*
 * Free one node and the adjacency arrays it owns.
 */
static void destroy_node_storage(HNSWNode *node)
{
    if (node == NULL)
    {
        return;
    }

    if (node->neighbors != NULL)
    {
        for (int level = 0; level <= node->max_level; level++)
        {
            free(node->neighbors[level].ids);
        }

        free(node->neighbors);
    }

    free(node);
}


/*
 * Public index destructor.
 */
void hnsw_destroy(HNSWIndex *index)
{
    if (index == NULL)
    {
        return;
    }

    for (int i = 0; i < index->count; i++)
    {
        destroy_node_storage(index->nodes[i]);
    }

    free(index->nodes);
    free(index);
}


/*
 * Configure how much search effort is used during insertion.
 * Set this before building an index for consistent experiments.
 */
int hnsw_set_ef_construction(HNSWIndex *index, int ef)
{
    if (index == NULL || ef < 1)
    {
        return 0;
    }

    index->ef_construction = ef;

    return 1;
}


/*
 * Remove references to a newly inserted node and release it
 * if insertion fails after the node has entered the index.
 *
 * This keeps the graph structurally consistent. If pruning
 * had already changed existing neighbor lists, those valid
 * changes are not reversed.
 */
static void rollback_insert(
    HNSWIndex *index,
    int node_id,
    int previous_entry_point,
    int previous_max_level
)
{
    if (index == NULL ||
        index->count <= 0 ||
        node_id != index->count - 1)
    {
        return;
    }

    for (int i = 0; i < index->count; i++)
    {
        HNSWNode *node = index->nodes[i];

        if (node == NULL || node->neighbors == NULL)
        {
            continue;
        }

        for (int level = 0; level <= node->max_level; level++)
        {
            NeighborList *list = &node->neighbors[level];

            while (neighbor_list_remove_id(list, node_id))
            {
                /* Remove any remaining reference to this ID. */
            }
        }
    }

    HNSWNode *inserted = index->nodes[node_id];

    index->nodes[node_id] = NULL;
    index->count--;

    destroy_node_storage(inserted);

    index->entry_point = previous_entry_point;
    index->max_level = previous_max_level;
}


/*
 * Find the closest node ID within the result heap.
 * The heap itself does not need to be sorted.
 */
static int nearest_result_id(const Heap *results)
{
    if (results == NULL || results->size <= 0)
    {
        return -1;
    }

    int best_id = -1;
    float best_distance = 0.0f;

    for (int i = 0; i < results->size; i++)
    {
        HeapItem item = results->items[i];

        if (best_id == -1 || item.distance < best_distance)
        {
            best_id = item.node_id;
            best_distance = item.distance;
        }
    }

    return best_id;
}


/*
 * Insert a vector into the HNSW index.
 *
 * Returns the newly assigned node ID on success,
 * or -1 on failure.
 */
int hnsw_insert(HNSWIndex *index, Vector vector)
{
    if (index == NULL ||
        index->M < 2 ||
        index->M > INT_MAX / 2 ||
        index->ef_construction < 1 ||
        index->count < 0 ||
        index->count == INT_MAX)
    {
        return -1;
    }

    if (index->count > 0 &&
        (index->entry_point < 0 ||
         index->entry_point >= index->count ||
         index->max_level < 0))
    {
        return -1;
    }

    int node_id = index->count;
    int new_level = random_level(index);

    HNSWNode *new_node = node_create(
        node_id,
        vector,
        new_level
    );

    if (new_node == NULL)
    {
        return -1;
    }

    /*
     * The first node has no existing graph to search.
     */
    if (index->count == 0)
    {
        if (!hnsw_add_node(index, new_node))
        {
            destroy_node_storage(new_node);
            return -1;
        }

        return node_id;
    }

    int previous_entry_point = index->entry_point;
    int previous_max_level = index->max_level;

    int current_entry = index->entry_point;

    /*
     * Phase 1:
     * Greedily navigate layers above the new node's level.
     * These layers help us approach the appropriate region.
     */
    for (int level = index->max_level;
         level > new_level;
         level--)
    {
        current_entry = greedy_search_layer(
            index,
            &vector,
            current_entry,
            level
        );

        if (current_entry < 0)
        {
            destroy_node_storage(new_node);
            return -1;
        }
    }

    /*
     * Store the node so its ID can be referenced by edges.
     * hnsw_add_node also updates the index's highest-level
     * entry point if this node is promoted above the old top.
     */
    if (!hnsw_add_node(index, new_node))
    {
        destroy_node_storage(new_node);
        return -1;
    }

    int highest_connection_level =
        (new_level < previous_max_level)
            ? new_level
            : previous_max_level;

    /*
     * Phase 2:
     * At every layer the new node belongs to, search for
     * candidates, select neighbors and create connections.
     */
    for (int level = highest_connection_level;
         level >= 0;
         level--)
    {
        Heap *candidates = hnsw_search_layer(
            index,
            &vector,
            current_entry,
            level,
            index->ef_construction
        );

        if (candidates == NULL)
        {
            rollback_insert(
                index,
                node_id,
                previous_entry_point,
                previous_max_level
            );

            return -1;
        }

        int nearest_id = nearest_result_id(candidates);

        if (nearest_id < 0)
        {
            heap_free(candidates);

            rollback_insert(
                index,
                node_id,
                previous_entry_point,
                previous_max_level
            );

            return -1;
        }

        int neighbor_limit =
            (level == 0) ? 2 * index->M : index->M;

        NeighborList selected = {NULL, 0, 0};

        int selection_ok = hnsw_select_neighbors(
            index,
            &vector,
            candidates,
            neighbor_limit,
            &selected
        );

        heap_free(candidates);

        if (!selection_ok)
        {
            rollback_insert(
                index,
                node_id,
                previous_entry_point,
                previous_max_level
            );

            return -1;
        }

        int connections_ok = 1;

        for (int i = 0; i < selected.count; i++)
        {
            if (!hnsw_connect_nodes(
                    index,
                    node_id,
                    selected.ids[i],
                    level))
            {
                connections_ok = 0;
                break;
            }
        }

        free(selected.ids);

        if (!connections_ok)
        {
            rollback_insert(
                index,
                node_id,
                previous_entry_point,
                previous_max_level
            );

            return -1;
        }

        /*
         * Carry the closest discovered candidate downward
         * as the starting point for the next layer.
         */
        current_entry = nearest_id;
    }

    return node_id;
}


/*
 * Perform a complete HNSW query:
 *
 * 1. Start at the highest-level entry point.
 * 2. Greedily navigate the upper layers.
 * 3. Perform best-first search at Layer 0.
 * 4. Retain the k nearest results.
 *
 * The returned max-heap contains up to k results.
 * Its root is the worst of those retained results.
 */
Heap *hnsw_search(
    const HNSWIndex *index,
    const Vector *query,
    int k,
    int ef_search
)
{
    if (index == NULL ||
        query == NULL ||
        index->nodes == NULL ||
        index->count <= 0 ||
        k <= 0 ||
        ef_search <= 0 ||
        index->entry_point < 0 ||
        index->entry_point >= index->count ||
        index->max_level < 0)
    {
        return NULL;
    }

    HNSWNode *entry = index->nodes[index->entry_point];

    if (entry == NULL ||
        entry->id != index->entry_point ||
        entry->max_level != index->max_level)
    {
        return NULL;
    }

    /*
     * We cannot return more results than there are nodes.
     */
    int result_limit =
        (k < index->count) ? k : index->count;

    /*
     * Search breadth must be at least k so that the
     * search can retain enough candidates for its output.
     */
    int effective_ef =
        (ef_search < result_limit)
            ? result_limit
            : ef_search;

    int current_entry = index->entry_point;

    /*
     * Phase 1: navigate from the highest layer down
     * through the upper layers using greedy search.
     *
     * Layer 0 is handled separately by best-first search.
     */
    for (int level = index->max_level; level > 0; level--)
    {
        current_entry = greedy_search_layer(
            index,
            query,
            current_entry,
            level
        );

        if (current_entry < 0)
        {
            return NULL;
        }
    }

    /*
     * Phase 2: perform broader best-first exploration
     * at Layer 0.
     */
    Heap *layer_results = hnsw_search_layer(
        index,
        query,
        current_entry,
        0,
        effective_ef
    );

    if (layer_results == NULL)
    {
        return NULL;
    }

    /*
     * Phase 3: reduce the ef-sized result set to
     * the requested k nearest neighbors.
     *
     * Both heaps use the same distance-based priority
     * rule, but this output heap is capped at k.
     */
    Heap *top_k = heap_create(result_limit, MAX_HEAP);

    if (top_k == NULL)
    {
        heap_free(layer_results);
        return NULL;
    }

    for (int i = 0; i < layer_results->size; i++)
    {
        HeapItem candidate = layer_results->items[i];

        if (candidate.node_id < 0 ||
            candidate.node_id >= index->count ||
            index->nodes[candidate.node_id] == NULL ||
            !isfinite(candidate.distance))
        {
            heap_free(layer_results);
            heap_free(top_k);
            return NULL;
        }

        /*
         * Until we have k results, keep adding candidates.
         */
        if (top_k->size < result_limit)
        {
            if (!heap_push(top_k, candidate))
            {
                heap_free(layer_results);
                heap_free(top_k);
                return NULL;
            }

            continue;
        }

        /*
         * Once full, compare the candidate against the
         * worst retained result, available at the max-heap root.
         */
        HeapItem worst;

        if (!heap_peek(top_k, &worst))
        {
            heap_free(layer_results);
            heap_free(top_k);
            return NULL;
        }

        if (candidate.distance < worst.distance)
        {
            HeapItem removed;

            if (!heap_pop(top_k, &removed) ||
                !heap_push(top_k, candidate))
            {
                heap_free(layer_results);
                heap_free(top_k);
                return NULL;
            }
        }
    }

    heap_free(layer_results);

    return top_k;
}
