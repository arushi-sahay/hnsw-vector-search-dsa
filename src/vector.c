#include "vector.h"
#include <math.h>

float euclidean_distance(const Vector *a, const Vector *b)
{
    float sum = 0.0f;

    for (int i = 0; i < DIM; i++)
    {
        float difference = a->values[i] - b->values[i];
        sum += difference * difference;
    }

    return sqrtf(sum);
}
