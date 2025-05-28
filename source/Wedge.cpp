/*
 * Wedge.cpp
 *
 *  Created on: 28.03.2015
 *      Author: swenzel
 */
#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge.h"
#include <iostream>
#include <iomanip>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
Wedge::Wedge(Precision angle, Precision zeroangle) : fSPhi(zeroangle), fDPhi(angle), fAlongVector1(), fAlongVector2()
{
  // check input
  VECGEOM_ASSERT(angle > 0.0 && angle <= kTwoPi);

  // initialize angles
  fAlongVector1.x() = std::cos(fSPhi);
  fAlongVector1.y() = std::sin(fSPhi);
  fAlongVector2.x() = std::cos(fSPhi + fDPhi);
  fAlongVector2.y() = std::sin(fSPhi + fDPhi);

  fNormalVector1.x() = -std::sin(fSPhi);
  fNormalVector1.y() = std::cos(fSPhi); // not the + sign
  fNormalVector2.x() = std::sin(fSPhi + fDPhi);
  fNormalVector2.y() = -std::cos(fSPhi + fDPhi); // note the - sign
}
}
}
