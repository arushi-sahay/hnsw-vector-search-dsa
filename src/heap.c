#include "heap.h"
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

Heap *heap_create(int capacity, HeapType type)
{
    if (type != MIN_HEAP && type != MAX_HEAP)
    {
        return NULL;
    }

    if (capacity < 1)
    {
        capacity = 4;
    }

    Heap *heap = malloc(sizeof(Heap));

    if (heap == NULL)
    {
        return NULL;
    }

    heap->items = malloc((size_t)capacity * sizeof(HeapItem));

    if (heap->items == NULL)
    {
        free(heap);
        return NULL;
    }

    heap->size = 0;
    heap->capacity = capacity;
    heap->type = type;

    return heap;
}

static int has_higher_priority(
    const Heap *heap,
    HeapItem a,
    HeapItem b
)
{
    if (heap->type == MIN_HEAP)
    {
        return a.distance < b.distance;
    }

    return a.distance > b.distance;
}

int heap_push(Heap *heap, HeapItem item)
{
    if (heap == NULL ||
        heap->size < 0 ||
        heap->capacity < 0 ||
        heap->size > heap->capacity ||
        (heap->capacity > 0 && heap->items == NULL) ||
        (heap->type != MIN_HEAP && heap->type != MAX_HEAP))
    {
        return 0;
    }

    if (heap->size == heap->capacity)
    {
        int new_capacity;

        if (heap->capacity == 0)
        {
            new_capacity = 4;
        }
        else
        {
            if (heap->capacity > INT_MAX / 2)
            {
                return 0;
            }

            new_capacity = heap->capacity * 2;
        }

        if ((size_t)new_capacity > SIZE_MAX / sizeof(HeapItem))
        {
            return 0;
        }

        HeapItem *new_items = realloc(
            heap->items,
            (size_t)new_capacity * sizeof(HeapItem)
        );

        if (new_items == NULL)
        {
            return 0;
        }

        heap->items = new_items;
        heap->capacity = new_capacity;
    }

    int index = heap->size;

    heap->items[index] = item;
    heap->size++;

    while (index > 0)
    {
        int parent = (index - 1) / 2;

        if (!has_higher_priority(
                heap,
                heap->items[index],
                heap->items[parent]))
        {
            break;
        }

        HeapItem temporary = heap->items[index];
        heap->items[index] = heap->items[parent];
        heap->items[parent] = temporary;

        index = parent;
    }

    return 1;
}


int heap_peek(const Heap *heap, HeapItem *item)
{
    if (heap == NULL ||
        item == NULL ||
        heap->items == NULL ||
        heap->size <= 0 ||
        heap->size > heap->capacity)
    {
        return 0;
    }

    *item = heap->items[0];

    return 1;
}


int heap_pop(Heap *heap, HeapItem *item)
{
    if (heap == NULL ||
        item == NULL ||
        heap->items == NULL ||
        heap->size <= 0 ||
        heap->size > heap->capacity)
    {
        return 0;
    }

    *item = heap->items[0];

    heap->size--;

    if (heap->size == 0)
    {
        return 1;
    }

    heap->items[0] = heap->items[heap->size];

    int index = 0;

    while (index < heap->size / 2)
    {
        int left = 2 * index + 1;
        int right = left + 1;
        int best = index;

        if (has_higher_priority(
                heap,
                heap->items[left],
                heap->items[best]))
        {
            best = left;
        }

        if (right < heap->size &&
            has_higher_priority(
                heap,
                heap->items[right],
                heap->items[best]))
        {
            best = right;
        }

        if (best == index)
        {
            break;
        }

        HeapItem temporary = heap->items[index];
        heap->items[index] = heap->items[best];
        heap->items[best] = temporary;

        index = best;
    }

    return 1;
}

int heap_is_empty(const Heap *heap)
{
    return heap == NULL || heap->size == 0;
}

void heap_free(Heap *heap)
{
    if (heap == NULL)
    {
        return;
    }

    free(heap->items);
    free(heap);
}
