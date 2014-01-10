#ifndef BOX_H
#define BOX_H

#include "Shape.h"
#include "LibraryGeneric.h"

class Box : public Shape {

private:

  Vector3D<double> dimensions;

public:

  Box(const Vector3D<double> dim, TransMatrix<double> const * const trans) {
    dimensions = dim;
    trans_matrix = trans;
    bounding_box = this;
  }

  Vector3D<double> const& Dimensions() const {
    return dimensions;
  }

  inline void SetCudaMatrix(TransMatrix<float> const * const trans_cuda) {
    trans_matrix_cuda = trans_cuda;
  }

  // Contains

  template <ImplType it>
  bool Contains(Vector3D<double> const& /*point*/) const;

  template <ImplType it>
  void Contains(SOA3D<double> const& /*points*/,
                bool* /*output*/) const;

  // DistanceToIn

  template <ImplType it>
  double DistanceToIn(Vector3D<double> const& /*point*/,
                      Vector3D<double> const& /*dir*/) const;

  template <ImplType it>
  void DistanceToIn(SOA3D<double> const& /*pos*/,
                    SOA3D<double> const& /*dir*/,
                    double* distance) const;

};

#endif /* BOX_H */