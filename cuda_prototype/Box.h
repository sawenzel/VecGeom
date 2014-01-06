#ifndef BOX_H
#define BOX_H

#include "LibraryGeneric.h"

class Box {

private:

  Vector3D<double> dimensions;
  TransMatrix const *trans_matrix;

public:

  #ifdef CXX_STD11
  Box(const Vector3D<double> dim_, TransMatrix const trans)
      : dimensions(dim), trans_matrix(trans) {}
  #else
  Box(const Vector3D<double> dim_, TransMatrix const * const trans) {
    dimensions = dim_;
    trans_matrix = trans;
  }
  #endif /* CXX_STD11 */

  template <ImplType it>
  void Contains(SOA3D<double> const& /*points*/,
                bool* /*output*/) const;

};

#endif /* BOX_H */