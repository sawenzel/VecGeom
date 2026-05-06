#ifndef VECGEOM_BASE_PRIOQUEUE_H_
#define VECGEOM_BASE_PRIOQUEUE_H_

#include "VecGeom/base/Assert.h"

/**
 * @brief A simple implementation of a small priority queue, compilable under CPU + GPU
 * Can be used in BVH traversal etc
 */
namespace vecgeom {

template <typename T, typename Priority, int MaxN = 1000>
struct PriorityQueue {
  struct Entry {
    T value;
    Priority priority;
  };
  // make sure to never initialize
  Entry data[MaxN];
  int size = 0;
  VECCORE_ATT_HOST_DEVICE void push(T val, Priority p)
  {
    int i = size++;
    VECGEOM_ASSERT(i < MaxN);
    data[i] = {val, p};
    // Sift up (min-heap on priority)
    while (i > 0) {
      int parent = (i - 1) >> 1;
      if (data[parent].priority <= data[i].priority) {
        break;
      }
      Entry tmp    = data[parent];
      data[parent] = data[i];
      data[i]      = tmp;
      i            = parent;
    }
  }
  VECCORE_ATT_HOST_DEVICE
  T pop()
  {
    T top = data[0].value;
    VECGEOM_ASSERT(size > 0);
    data[0] = data[--size];
    // Sift down
    int i = 0;
    while (true) {
      int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
      if (l < size && data[l].priority < data[smallest].priority) {
        smallest = l;
      }
      if (r < size && data[r].priority < data[smallest].priority) {
        smallest = r;
      }
      if (smallest == i) {
        break;
      }
      Entry tmp      = data[smallest];
      data[smallest] = data[i];
      data[i]        = tmp;
      i              = smallest;
    }
    return top;
  }
  VECCORE_ATT_HOST_DEVICE T peek() const { return data[0].value; }
  VECCORE_ATT_HOST_DEVICE bool empty() const { return size == 0; }
  VECCORE_ATT_HOST_DEVICE void clear() { size = 0; }
  VECCORE_ATT_HOST_DEVICE Priority peek_priority() const { return data[0].priority; }
  VECCORE_ATT_HOST_DEVICE void trim(Priority threshold)
  {
    while (size > 0 && data[size - 1].priority > threshold)
      --size;
  }
};

} // namespace vecgeom

#endif
