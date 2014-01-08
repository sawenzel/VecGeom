#ifndef BOX_H
#define BOX_H

#include "Shape.h"
#include "LibraryGeneric.h"

struct BoxParameters : public ShapeParameters {
  BoxParameters(Vector3D<double> const &dim) : dimensions(dim) {}
  Vector3D<double> dimensions;
};

class Box : public Shape {

public:

  typedef BoxParameters ParameterType;

  Box(const Vector3D<double> dim, TransMatrix<double> const * const trans) {
    parameters = new BoxParameters(dim);
    trans_matrix = trans;
    bounding_box = this;
  }

  ~Box() {
    delete parameters;
  }

  Vector3D<double> const& Dimensions() const {
    return ((BoxParameters*)parameters)->dimensions;
  }

  inline void SetCudaMatrix(TransMatrix<float> const * const trans_cuda) {
    trans_matrix_cuda = trans_cuda;
  }

};

#endif /* BOX_H */