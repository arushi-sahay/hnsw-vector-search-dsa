#include "node.h"
#include <stdlib.h>

HNSWNode *node_create(int id, Vector vector, int max_level)
{
    HNSWNode *node = malloc(sizeof(HNSWNode));

    if (node == NULL)
    {
        return NULL;
    }

    node->id = id;
    node->vector = vector;
    node->max_level = max_level;

    node->neighbors = malloc((max_level + 1) * sizeof(NeighborList));

    if (node->neighbors == NULL)
    {
        free(node);
        return NULL;
    }

    for (int level = 0; level <= max_level; level++)
    {
        node->neighbors[level].ids = NULL;
        node->neighbors[level].count = 0;
        node->neighbors[level].capacity = 0;
    }

    return node;
}

int neighbor_list_add(NeighborList *list, int neighbor_id)
{
    if (list->count == list->capacity)
    {
        int new_capacity;

        if (list->capacity == 0)
        {
            new_capacity = 4;
        }
        else
        {
            new_capacity = list->capacity * 2;
        }

        int *new_ids = realloc(
            list->ids,
            new_capacity * sizeof(int)
        );

        if (new_ids == NULL)
        {
            return 0;
        }

        list->ids = new_ids;
        list->capacity = new_capacity;
    }

    list->ids[list->count] = neighbor_id;
    list->count++;

    return 1;
}
