#ifndef BOX_H
#define BOX_H

#include "LibraryGeneric.h"

class Box {

private:

  Vector3D<double> dimensions;
  TransMatrix<double> const *trans_matrix;
  TransMatrix<float> const *trans_matrix_cuda;

public:

  #ifdef STD_CXX11
  Box(const Vector3D<double> dim, TransMatrix<double> const * const trans)
      : dimensions(dim), trans_matrix(trans) {}
  #else
  Box(const Vector3D<double> dim, TransMatrix<double> const * const trans) {
    dimensions = dim;
    trans_matrix = trans;
  }
  #endif /* STD_CXX11 */

  template <ImplType it>
  void Contains(SOA3D<double> const& /*points*/,
                bool* /*output*/) const;

  template <ImplType it>
  void DistanceToIn(SOA3D<double> const& /*pos*/, SOA3D<double> const& /*dir*/,
                    double* /*distance*/) const;

  void SetCudaMatrix(TransMatrix<float> const * const trans_cuda) {
    trans_matrix_cuda = trans_cuda;
  }

};

#endif /* BOX_H */