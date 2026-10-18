#ifndef HEAP_H
#define HEAP_H

typedef enum
{
    MIN_HEAP,
    MAX_HEAP
} HeapType;

typedef struct
{
    float distance;
    int node_id;
} HeapItem;

typedef struct
{
    HeapItem *items;
    int size;
    int capacity;
    HeapType type;
} Heap;

Heap *heap_create(int capacity, HeapType type);

int heap_push(Heap *heap, HeapItem item);

int heap_peek(const Heap *heap, HeapItem *item);

int heap_pop(Heap *heap, HeapItem *item);

int heap_is_empty(const Heap *heap);

void heap_free(Heap *heap);

#endif
