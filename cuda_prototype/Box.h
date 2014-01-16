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

  // Contains

  virtual bool Contains(Vector3D<double> const& /*point*/) const;

  virtual void Contains(SOA3D<double> const& /*points*/,
                        bool* /*output*/) const;

  // DistanceToIn

  virtual double DistanceToIn(Vector3D<double> const& /*point*/,
                              Vector3D<double> const& /*dir*/,
                              double const step_max) const;

  virtual void DistanceToIn(SOA3D<double> const& /*pos*/,
                            SOA3D<double> const& /*dir*/,
                            double const *step_max,
                            double* distance) const;

};

#endif /* BOX_H */