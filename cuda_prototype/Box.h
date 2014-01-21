#ifndef BOX_H
#define BOX_H

#include "Shape.h"
#include "LibraryGeneric.h"

struct BoxParameters {
  Vector3D<double> dimensions;
  #ifdef STD_CXX11
  BoxParameters(Vector3D<double> const& dim) : dimensions(dim) {}
  #else
  BoxParameters(Vector3D<double> const& dim) { dimensions = dim; }
  #endif
};

class Box : public Shape {

private:

  BoxParameters const *parameters;

public:

  #ifdef STD_CXX11
  Box(BoxParameters const * const params,
      TransMatrix<double> const * const trans)
      : Shape(trans, this), parameters(params) {}
  #else
  Box(BoxParameters const * const params,
      TransMatrix<double> const * const trans)
      : Shape(trans, this) {
    parameters = params;
  }
  #endif

  Vector3D<double> const& Dimensions() const { return parameters->dimensions; }
  double const& X() const { return parameters->dimensions[0]; }
  double const& Y() const { return parameters->dimensions[1]; }
  double const& Z() const { return parameters->dimensions[2]; }

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