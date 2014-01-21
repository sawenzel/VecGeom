#include "LibraryCilk.h"
#include "Kernel.h"
#include "Box.h"

const CilkBool ImplTraits<kCilk>::kTrue = CilkBool(true);
const CilkBool ImplTraits<kCilk>::kFalse = CilkBool(false);
const CilkFloat ImplTraits<kCilk>::kZero = CilkFloat(0);

void Box::DistanceToIn(SOA3D<double> const &pos,
                       SOA3D<double> const &dir,
                       double const *steps_max,
                       double *distance) const {

  const int size = pos.size();
  const Vector3D<CilkFloat> pos_cilk(
    CilkFloat(pos.x(), size, false),
    CilkFloat(pos.y(), size, false),
    CilkFloat(pos.z(), size, false)
  );
  const Vector3D<CilkFloat> dir_cilk(
    CilkFloat(dir.x(), size, false),
    CilkFloat(dir.y(), size, false),
    CilkFloat(dir.z(), size, false)
  );
  const CilkFloat step_max(steps_max, size, false);
  CilkFloat output(distance, size, false);
  kernel::box::DistanceToIn<kCilk>(
    dimensions, trans_matrix, pos_cilk, dir_cilk, step_max, output
  );

}