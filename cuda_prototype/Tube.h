#ifndef TUBE_H
#define TUBE_H

#include "Shape.h"
#include "LibraryGeneric.h"
#include "Box.h"

struct TubeParameters {
  double r_min, r_max;
  double z;
  double phi_min, phi_max;
  #ifdef STD_CXX11
  TubeParameters(const double r_min_, const double r_max_, const double z_,
                 const double phi_min_, const double phi_max_)
      : r_min(r_min_), r_max(r_max_), z(z_), phi_min(phi_min_),
        phi_max(phi_max_) {}
  #else
  TubeParameters(const double r_min_, const double r_max_, const double z_,
                 const double phi_min_, const double phi_max_) {
    r_min = r_min_;
    r_max = r_max_;
    z = z_;
    phi_min = phi_min_;
    phi_max = phi_max_;
  }
  #endif
};

class Tube : public Shape {

private:

  TubeParameters const *parameters;

public:

  #ifdef STD_CXX11
  Tube(TubeParameters const * const params,
       TransMatrix<double> const * const matrix)
      : Shape(matrix, nullptr), parameters(params) {}
  #else
  Tube(TubeParameters const * const params,
       TransMatrix<double> const * const matrix)
      : Shape(matrix, NULL) {
    parameters = params;
  } 
  #endif

  const double& RMin()   const { return parameters->r_min;   }
  const double& RMax()   const { return parameters->r_max;   }
  const double& Z()      const { return parameters->z;       }
  const double& PhiMin() const { return parameters->phi_min; }
  const double& PhiMax() const { return parameters->phi_max; }
  
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

#endif /* TUBE_H */