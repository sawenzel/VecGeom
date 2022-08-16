#ifndef VECGEOM_ALGORITHMS_H
#define VECGEOM_ALGORITHMS_H

#include <cmath>
#include <limits>
#include "VecCore/Limits.h"

namespace vecgeom {
namespace algo {

///< A variant of the quicksort algorithm, avoiding recursiveness by using a limited-size
///< stack as 'max_levels'. Upon exhausing the stack sorting can fail, so the return value must be
///< checked. The maximum levels needed for sorting an array of size `len` is approximately:
///<     max_levels ~ 3 * log10(len)
template <typename Type, int max_levels = 32>
int quickSort(Type const *arr, size_t elements, size_t *sorted)
{
  ///< Implementation credits go to: https://stackoverflow.com/a/55011578

  size_t beg[max_levels], end[max_levels], L, R;
  int i = 0;

  for (auto j = 0; j < elements; ++j)
    sorted[j] = j;

  beg[0] = 0;
  end[0] = elements;
  while (i >= 0) {
    L = beg[i];
    R = end[i];
    if (R - L > 1) {
      size_t M   = L + ((R - L) >> 1);
      size_t piv = sorted[M];
      sorted[M]  = sorted[L];
      if (i == max_levels - 1) return -1;
      R--;
      while (L < R) {
        while (arr[sorted[R]] >= arr[piv] && L < R)
          R--;
        if (L < R) sorted[L++] = sorted[R];
        while (arr[sorted[L]] <= arr[piv] && L < R)
          L++;
        if (L < R) sorted[R--] = sorted[L];
      }
      sorted[L] = piv;
      M         = L + 1;
      while (L > beg[i] && arr[sorted[L - 1]] == arr[piv])
        L--;
      while (M < end[i] && arr[sorted[M]] == arr[piv])
        M++;
      if (L - beg[i] > end[i] - M) {
        beg[i + 1] = M;
        end[i + 1] = end[i];
        end[i++]   = L;
      } else {
        beg[i + 1] = beg[i];
        end[i + 1] = L;
        beg[i++]   = M;
      }
    } else {
      i--;
    }
  }
  return 0;
}

} // end namespace algo
} // end namespace vecgeom

#endif
