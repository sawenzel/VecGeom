#ifndef VECGEOM_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_SHAPEIMPLEMENTATIONHELPER_H_

#include "Global.h"

template <class Shape, class Specialization>
struct ShapeImplementationHelper : public Shape {

public:

  ShapeImplementationHelper(
      typename Shape::UnplacedShape_t const *const unplacedShape)
      : Shape(unplacedShape) {}

  virtual bool Inside(Vector3D<double> const &point) const {
    bool output;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );
    return output;
  }

#ifdef VECGEOM_VC
  virtual void Inside(double const *const *const points, const int n,
                      bool *const output) const {
    for (int i = 0; i < n; i += VcDouble::Size) {
      const Vector3D<VcDouble> point(
        VcDouble(&points[0][i]),
        VcDouble(&points[1][i]),
        VcDouble(&points[2][i])
      );
      VcBool output_vc;
      Specialization::template Inside<kVc>(
        *this->GetUnplacedVolume(),
        point,
        output_vc
      );
      for (int j = 0; j < n; ++j) {
        output[i+j] = output_vc[j];
      }
    }
  }
#else // Scalar looper
  virtual void Inside(double const *const *const points, const int n,
                      bool *const output) const {
    for (int i = 0; i < n; ++i) {
      const Vector3D<double> point(points[0][i], points[1][i], points[2][i]);
      Specialization::template Inside<kScalar>(
        *this->GetUnplacedVolume(),
        point,
        output[i]
      );
    }
  }
#endif


};

#endif // VECGEOM_SHAPEIMPLEMENTATIONHELPER_H_