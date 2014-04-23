#ifndef VECGEOM_SHAPEIMPLEMENTATION_H_
#define VECGEOM_SHAPEIMPLEMENTATION_H_

#include "PlacedVolume.h"
#include "Kernel.h"

template <class PlacedType, class Specialization>
class ShapeImplementation : public PlacedVolume {

public:

  virtual bool Inside(double const point[3]) const {
    bool output;
    PlacedType const *const placed = static_cast<PlacedType const*>(this);
    placed->template InsideDispatch<kScalar, Specialization>(
      point,
      output
    );
    return output;
  }

#ifdef VECGEOM_VC
  virtual void Inside(const double points[3][VcDouble::Size],
                      bool output[VcDouble::Size]) const {
    VcBool output_vc;
    VcDouble points_vc[3] = {VcDouble(points[0]), VcDouble(points[1]),
                             VcDouble(points[2])};
    PlacedType const *const placed = static_cast<PlacedType const*>(this);
    placed->template InsideDispatch<kVc, Specialization>(
      points_vc,
      output_vc
    );
    for (int i = 0; i < VcDouble::Size; ++i) output[i] = output_vc[i];
  }
#endif

};

#endif // VECGEOM_SHAPEIMPLEMENTATION_H_