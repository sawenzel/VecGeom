/*
 * PhiWedgeTest.cpp
 *
 *  Created on: 09.10.2014
 *      Author: swenzel
 */



#include "volumes/Wedge.h"
#include "backend/Backend.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace vecgeom;

typedef Vector3D<Precision> Vector3D_t;

__attribute__((noinline))
// this function is just here for inspection of assembly output
int foo( Wedge const & wedge, Vector3D_t const & point ){
    return wedge.Inside<kScalar>( point );
}

int main()
{


    {
        // test reported by Raman Segal
        Wedge wedge( 3.*kPi/2., kPi/6.);
        assert( wedge.IsOnSurface1( Vector3D_t(
                       4.589551141500899, 2.649778586424563, 4.573258549707597 ) ) );
        assert( ! wedge.IsOnSurface2( Vector3D_t(
                              4.589551141500899, 2.649778586424563, 4.573258549707597 ) ) );

        assert( wedge.Inside<kScalar>( Vector3D_t(
                4.589551141500899, 2.649778586424563, 4.573258549707597 ) ) == EInside::kSurface );

        // the opposite point should be inside
        assert( wedge.Contains<kScalar>( Vector3D_t(
                       -4.589551141500899, -2.649778586424563, 4.573258549707597 ) ) );
        assert( wedge.Inside<kScalar>( Vector3D_t(
                       -4.589551141500899, -2.649778586424563, 4.573258549707597 ) ) == EInside::kInside );

        assert( wedge.Inside<kScalar>( Vector3D_t( 0., 0., 0.) ) == EInside::kSurface );
    }


    // test a wedge < kPi
    {
    Wedge wedge(kPi/3.);

    // tests on contains
    assert( !wedge.Contains<kScalar> ( Vector3D_t(1., -2.*kTolerance, 0.) ) );
    assert( wedge.Contains<kScalar> ( Vector3D_t(1., 0. ,0.) ) );
    assert( !wedge.Contains<kScalar> ( Vector3D_t(1., -1.*kTolerance, 0.) ) );
    assert( !wedge.Contains<kScalar>  ( Vector3D_t(-1, -1., 0.) ) );
    assert( wedge.Contains<kScalar> ( Vector3D_t(std::cos(kPi/3.), std::sin(kPi/3.), 0. )));
    assert( !wedge.Contains<kScalar> ( Vector3D_t(std::cos(kPi/3.+kTolerance), std::sin(kPi/3.+kTolerance), 0. )));
    assert( wedge.Contains<kScalar> ( Vector3D_t(std::cos(kPi/6.), std::sin(kPi/6.), 0.) ) );
    assert( !wedge.Contains<kScalar> ( Vector3D_t(std::cos(kPi/2.), std::sin(kPi/2.), 0.) ) );

    assert( !wedge.Contains<kScalar>( Vector3D_t(-1.,0.,0.) ) );

    // tests on inside
    assert( wedge.Inside<kScalar> ( Vector3D_t(1., 0. ,0.) ) == EInside::kSurface );
    assert( wedge.Inside<kScalar> ( Vector3D_t(1., -0.5*kTolerance ,0.) ) == EInside::kSurface );
    assert( wedge.Inside<kScalar> ( Vector3D_t(1., 0.5*kTolerance ,0.) ) == EInside::kSurface );
    assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(kPi/3.), std::sin(kPi/3.), 0. ) ) == EInside::kSurface );
    assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(kPi/3.+kTolerance), std::sin(kPi/3.+kTolerance), 0. ) ) == EInside::kOutside );
    assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(kPi/6.), std::sin(kPi/6.), 0.) ) == EInside::kInside );
    assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(kPi/2.), std::sin(kPi/2.), 0.) ) == EInside::kOutside );
    }

    // test a wedge > kPi
       {
           Precision angle =kPi + kPi/3.;
           Wedge wedge(angle);

       // tests on contains
       assert( !wedge.Contains<kScalar> ( Vector3D_t(1., -2.*kTolerance, 0.) ) );
       assert( wedge.Contains<kScalar> ( Vector3D_t(1., 0. ,0.) ) );
       assert( !wedge.Contains<kScalar> ( Vector3D_t(1., -1.*kTolerance, 0.) ) );
       assert( wedge.Contains<kScalar> ( Vector3D_t(std::cos(angle/3.), std::sin(angle/3.), 0. )));
       assert( !wedge.Contains<kScalar> ( Vector3D_t(std::cos(angle+kTolerance), std::sin(angle+kTolerance), 0. )));
       assert( wedge.Contains<kScalar> ( Vector3D_t(std::cos(angle/6.), std::sin(angle/6.), 0.) ) );
       assert( !wedge.Contains<kScalar> ( Vector3D_t(std::cos(angle + 0.1), std::sin(angle + 0.1), 0.) ) );

       // tests on inside
       assert( wedge.Inside<kScalar> ( Vector3D_t(1., 0. ,0.) ) == EInside::kSurface );
       assert( wedge.Inside<kScalar> ( Vector3D_t(1., -0.5*kTolerance ,0.) ) == EInside::kSurface );
       assert( wedge.Inside<kScalar> ( Vector3D_t(1., 0.5*kTolerance ,0.) ) == EInside::kSurface );
       assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(angle), std::sin(angle), 0. ) ) == EInside::kSurface );
       assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(angle+kTolerance), std::sin(angle+kTolerance), 0. ) ) == EInside::kOutside );
       assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(angle/6.), std::sin(angle/6.), 0.) ) == EInside::kInside );
       assert( wedge.Inside<kScalar> ( Vector3D_t(std::cos(angle + 0.1), std::sin(angle + 0.1), 0.) ) == EInside::kOutside );
       }
    return 0;
}
