#ifndef VECTOR_H
#define VECTOR_H

/*
 * Default dimension for the educational tests.
 * The benchmark can override this using -DDIM=64.
 */
#ifndef DIM
#define DIM 4
#endif

typedef struct
{
    float values[DIM];
} Vector;

float euclidean_distance(const Vector *a, const Vector *b);

#endif
