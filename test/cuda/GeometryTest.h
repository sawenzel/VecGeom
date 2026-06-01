// Author: Stephan Hageboeck, CERN, 2021
// Shared C++/Cuda infrastructure for a geometry test.

#include "VecGeom/volumes/PlacedVolume.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include "VecGeom/base/Assert.h"

namespace vecgeom {
VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_HOST_FORWARD_DECLARE(class VPlacedVolume;);
} // namespace vecgeom

struct GeometryInfo {
  unsigned int depth                                                    = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().id()) id              = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().GetChildId()) childId = 0;
  decltype(std::declval<vecgeom::VPlacedVolume>().GetCopyNo()) copyNo   = 0;
  decltype(std::declval<vecgeom::LogicalVolume>().id()) logicalId       = 0;
  vecgeom::Transformation3D trans;
  // Don't use Vector3D which changes layout from device to host in vector mode
  // so cudaMemcpy gets wrong data
  vecgeom::Precision amin[3];
  vecgeom::Precision amax[3];
  vecgeom::Precision unplacedSafety = 0.;

  GeometryInfo() = default;

  template <typename Vol_t>
  VECCORE_ATT_HOST_DEVICE GeometryInfo(unsigned int theDepth, Vol_t &vol)
      : depth(theDepth), id(vol.id()), childId(vol.GetChildId()), copyNo(vol.GetCopyNo()),
        trans{*vol.GetTransformation()}
  {
    logicalId           = vol.GetLogicalVolume()->id();
    const auto unplaced = vol.GetUnplacedVolume();
    VECGEOM_ASSERT(unplaced);

    vecgeom::Vector3D<vecgeom::Precision> aminv, amaxv;
    unplaced->GetBBox(aminv, amaxv);
    VECGEOM_ASSERT((amaxv - aminv).Mag() > 0 && "Bounding box size must be positive");
    for (auto i = 0; i < 3; ++i) {
      amin[i] = aminv[i];
      amax[i] = amaxv[i];
    }

    const vecgeom::Vector3D<vecgeom::Precision> testPoint{1., 2., -3.};
    unplacedSafety = unplaced->Contains(testPoint) ? unplaced->SafetyToOut(testPoint) : unplaced->SafetyToIn(testPoint);
  }

  bool operator==(const GeometryInfo &rhs) const
  {
    bool same_bbox = amin[0] == rhs.amin[0] && amin[1] == rhs.amin[1] && amin[2] == rhs.amin[2] &&
                     amax[0] == rhs.amax[0] && amax[1] == rhs.amax[1] && amax[2] == rhs.amax[2];
    const auto safetyScale =
        std::max(vecgeom::Precision(1.), std::max(std::fabs(unplacedSafety), std::fabs(rhs.unplacedSafety)));
    const bool correctSafety = std::fabs(unplacedSafety - rhs.unplacedSafety) <=
                               safetyScale * 256. * std::numeric_limits<vecgeom::Precision>::epsilon();
    return depth == rhs.depth && id == rhs.id && childId == rhs.childId && copyNo == rhs.copyNo &&
           logicalId == rhs.logicalId && trans == rhs.trans && same_bbox && correctSafety;
  }

  void print() const
  {
    printf("depth: %d, id: %d, childId: %d, copyNo: %d, logicalId: %d\n", depth, id, childId, copyNo, logicalId);
    trans.Print();
    printf("\namin: (%12.10E, %12.10E, %12.10E)  amax: (%12.10E, %12.10E, %12.10E)\n", amin[0], amin[1], amin[2],
           amax[0], amax[1], amax[2]);
    printf("unplaced safety: %15.13f\n", unplacedSafety);
  }
};

std::vector<GeometryInfo> visitDeviceGeometry(const vecgeom::cuda::VPlacedVolume *volume, std::size_t numVols,
                                              std::size_t stackCapacity);
