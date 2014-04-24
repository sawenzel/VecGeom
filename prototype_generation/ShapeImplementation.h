#ifndef VECGEOM_SHAPEIMPLEMENTATION_H_
#define VECGEOM_SHAPEIMPLEMENTATION_H_

#include "PlacedVolume.h"
#include "Kernel.h"

#define VECGEOM_INSIDE_IMPLEMENTATION \
VECGEOM_CUDA_HEADER_BOTH \
virtual bool Inside(const double point[3]) const { \
  return Implementation::Inside(point); \
} \
virtual void Inside(const double points[3][VcDouble::Size], \
                    bool output[VcDouble::Size]) const { \
  Implementation::Inside(points, output); \
}

template <class PlacedType, class Specialization>
class ShapeImplementation {

private:

  PlacedType const *const deriving_;

public:

  ShapeImplementation(PlacedType const *const deriving) : deriving_(deriving) {}

  bool Inside(double const point[3]) const {
    bool output;
    deriving_->template InsideDispatch<kScalar, Specialization>(
      point,
      output
    );
    return output;
  }

#ifdef VECGEOM_VC
  void Inside(const double points[3][VcDouble::Size],
                      bool output[VcDouble::Size]) const {
    VcBool output_vc;
    VcDouble points_vc[3] = {VcDouble(points[0]), VcDouble(points[1]),
                             VcDouble(points[2])};
    deriving_->template InsideDispatch<kVc, Specialization>(
      points_vc,
      output_vc
    );
    for (int i = 0; i < VcDouble::Size; ++i) output[i] = output_vc[i];
  }
#endif

};

#endif // VECGEOM_SHAPEIMPLEMENTATION_H_