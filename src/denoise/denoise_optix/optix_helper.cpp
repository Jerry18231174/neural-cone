#include "optix_helper.h"

bool operator==(const int3 &a, const int3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator!=(const int3 &a, const int3 &b) {
    return !(a == b);
}