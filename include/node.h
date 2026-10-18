#ifndef NODE_H
#define NODE_H

#include "vector.h"

typedef struct
{
    int *ids;
    int count;
    int capacity;
} NeighborList;

typedef struct
{
    int id;
    Vector vector;

    int max_level;

    NeighborList *neighbors;
} HNSWNode;

HNSWNode *node_create(int id, Vector vector, int max_level);
int neighbor_list_add(NeighborList *list, int neighbor_id);

#endif
