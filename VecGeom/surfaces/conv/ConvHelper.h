#ifndef VECGEOM_SURFACE_CONVHELPER_H_
#define VECGEOM_SURFACE_CONVHELPER_H_

#include <VecGeom/surfaces/conv/Builder.h>
#include <VecGeom/surfaces/Model.h>

namespace vgbrep {
namespace conv {

using Vector3D = vecgeom::Vector3D<vecgeom::Precision>;

/// @brief Helper function to check whether three vectors are right-sided
/// @param v1 vector 1
/// @param v2 vector 2
/// @param v3 vector 3
/// @return whether the three vectors are right-sided
bool IsRightSided(Vector3D v1, Vector3D v2, Vector3D v3)
{
  vecgeom::Precision dot = (v1[0] - v2[0]) * (v3[1] - v2[1]) - (v1[1] - v2[1]) * (v3[0] - v2[0]);
  return (dot < -vecgeom::kTolerance) ? false : true;
};

/// @brief Helper function to check whether a section is convex or concave
/// @tparam Container structure holding the data for z and r
/// @param zArray Points in z for all sections
/// @param rArray Points in r for all sections
/// @param convex whether to check for convexity or concavity
/// @param nsec number of section to be checked
/// @return if the section is convex/concave depending on the convex input parameter
template <typename Container>
bool IsConvexConcave(const int iSect, Container &zArray, Container &rArray, const bool convex, const int nsec)
{
  if (nsec == 2) return true;
  // bottom section, need to check only in the upper direction
  if (iSect == 0) {
    int i = iSect;
    int j = (i + 1);
    int k = (i + 2);
    // if the top of the this section and the bottom of next section are the same, we need to check the next point
    if (rArray[j] == rArray[k]) k++;
    if (convex ^ IsRightSided({rArray[i], zArray[i], 0.}, {rArray[j], zArray[j], 0.}, {rArray[k], zArray[k], 0.})) {
      return false;
    }
  }

  // mid section, need to check both in lower and in the upper direction
  if (iSect > 0 && iSect < nsec - 2) {
    int i = iSect - 1;
    int j = (i + 1);
    int k = (i + 2);
    // if the top of the previous section and the bottom of this section are the same, we need to check the previous
    // point
    if (rArray[i] == rArray[j]) i--;
    if (convex ^ IsRightSided({rArray[i], zArray[i], 0.}, {rArray[j], zArray[j], 0.}, {rArray[k], zArray[k], 0.})) {
      return false;
    }
    i = iSect;
    j = (i + 1);
    k = (i + 2);
    // if the top of the this section and the bottom of next section are the same, we need to check the next point
    if (rArray[j] == rArray[k]) k++;
    if (convex ^ IsRightSided({rArray[i], zArray[i], 0.}, {rArray[j], zArray[j], 0.}, {rArray[k], zArray[k], 0.})) {
      return false;
    }
  }

  // top section, need to check only in the lower direction
  if (iSect == nsec - 2) {
    int i = iSect - 1;
    int j = (i + 1);
    int k = (i + 2);
    // if the top of the previous section and the bottom of this section are the same, we need to check the previous
    // point
    if (rArray[i] == rArray[j]) i--;
    if (convex ^ IsRightSided({rArray[i], zArray[i], 0.}, {rArray[j], zArray[j], 0.}, {rArray[k], zArray[k], 0.})) {
      return false;
    }
  }

  return true;
};

} // namespace conv
} // namespace vgbrep
#endif
