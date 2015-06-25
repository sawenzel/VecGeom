/*
 * Wedge.cpp
 *
 *  Created on: 28.03.2015
 *      Author: swenzel
 */
#include "base/Global.h"
#include "volumes/Wedge.h"
#include <iostream>
#include <iomanip>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
        Wedge::Wedge( Precision angle, Precision zeroangle ) :
            // fSPhi(zeroangle),
            fDPhi(angle), fAlongVector1(), fAlongVector2() {
            // check input
            Assert( angle > 0., " wedge angle has to be larger than zero " );
            //
           // Assert( angle <= kTwoPi, "wedge angle is larger than 2*Pi; Are you using radians?" );
            if( ! ( angle <= kTwoPi + vecgeom::kTolerance ) ){
#ifndef VECGEOM_NVCC
                std::cerr << std::setprecision(20) << "\n";
                std::cerr << "wedge angle is larger than 2PI: "
                          << angle << " "
                          << kTwoPi-angle << "\n";
#else
                printf("\nwedge angle is larger than 2PI: angle=%f 2pi-angle=%f\n", angle, kTwoPi-angle);
#endif
            }

            // initialize angles
            fAlongVector1.x() = std::cos(zeroangle);
            fAlongVector1.y() = std::sin(zeroangle);
            fAlongVector2.x() = std::cos(zeroangle+angle);
            fAlongVector2.y() = std::sin(zeroangle+angle);

            fNormalVector1.x() = -std::sin(zeroangle);
            fNormalVector1.y() = std::cos(zeroangle);  // not the + sign
            fNormalVector2.x() =  std::sin(zeroangle+angle);
            fNormalVector2.y() = -std::cos(zeroangle+angle); // note the - sign
        }

}
}



