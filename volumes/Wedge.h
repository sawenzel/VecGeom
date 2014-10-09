/*
 * Wedge.h
 *
 *  Created on: 09.10.2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_WEDGE_H_
#define VECGEOM_VOLUMES_WEDGE_H_

#include "base/Global.h"
#include "volumes/kernel/GenericKernels.h"

namespace VECGEOM_NAMESPACE
{

/**
 * A class representing a wedge which is represented by an angle. It
 * can be used to divide 3D spaces or to clip wedges from solids.
 * The wedge has an "inner" and "outer" side. For an angle = 180 degree, the wedge is essentially
 * an ordinary halfspace.
 *
 * Idea: should have Unplaced and PlacedWegdes, should have specializations
 * for "PhiWegde" and "ThetaWegde" which are used in symmetric
 * shapes such as tubes or spheres.
 *
 * Note: This class is meant as an auxiliary class so it is a bit outside the ordinary volume
 * hierarchy.
 *
 */
class Wedge{

    private:
        Precision fAngle; // angle representing/defining the wedge
        Vector3D<Precision> fAlongVector1;
        Vector3D<Precision> fAlongVector2;

    public:
        Wedge( Precision angle ) : fAngle(angle), fAlongVector1(), fAlongVector2() {
            // check input
            Assert( angle > 0., " wedge angle has to be larger than zero " );

            // initialize angles
            fAlongVector1.x() = 1.;
            fAlongVector1.y() = 0.;
            fAlongVector2.x() = std::cos(angle);
            fAlongVector2.y() = std::sin(angle);
        }

        ~Wedge(){}

        // very important:
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::bool_v Contains( Vector3D<typename Backend::precision_v> const& point ) const;

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::inside_v Inside( Vector3D<typename Backend::precision_v> const& point ) const;

    private:
        // this could be useful to be public such that other shapes can directly
        // use completelyinside + completelyoutside

        template<typename Backend, bool ForInside>
        VECGEOM_CUDA_HEADER_BOTH
        void GenericKernelForContainsAndInside(
                Vector3D<typename Backend::precision_v> const &localPoint,
                typename Backend::bool_v &completelyinside,
                typename Backend::bool_v &completelyoutside) const;

};

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::inside_v Wedge::Inside( Vector3D<typename Backend::precision_v> const& point ) const
    {
        typedef typename Backend::bool_v      Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(
              point, completelyinside, completelyoutside);
        typename Backend::inside_v  inside=EInside::kSurface;
        MaskedAssign(completelyoutside, EInside::kOutside, &inside);
        MaskedAssign(completelyinside, EInside::kInside, &inside);
    }

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v Wedge::Contains( Vector3D<typename Backend::precision_v> const& point ) const
    {
        typedef typename Backend::bool_v Bool_t;
        Bool_t unused;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend, false>(
           point, unused, outside);
        return !outside;
    }

    // Implementation follows
    template<typename Backend, bool ForInside>
    VECGEOM_CUDA_HEADER_BOTH
    void Wedge::GenericKernelForContainsAndInside(
                Vector3D<typename Backend::precision_v> const &localPoint,
                typename Backend::bool_v &completelyinside,
                typename Backend::bool_v &completelyoutside) const
    {
        typedef typename Backend::precision_v Float_t;

       // this part of the code assumes some symmetry knowledge and is currently only
        // correct for a PhiWedge assumed to be aligned along the z-axis.
        Float_t x = localPoint.x();
        Float_t y = localPoint.y();
        Float_t startx = fAlongVector1.x( );
        Float_t starty = fAlongVector1.y( );
        Float_t endx = fAlongVector2.x( );
        Float_t endy = fAlongVector2.y( );

        Float_t startCheck = (-x*starty + y*startx);
        Float_t endCheck   = (-endx*y   + endy*x);

        // TODO: I think we need to treat the tolerance as a phi - tolerance
        // this will complicate things a little bit
        completelyoutside = startCheck < MakeMinusTolerant<ForInside>(0.);
        if(ForInside)
            completelyinside = startCheck > MakePlusTolerant<ForInside>(0.);

        if(fAngle<kPi) {
            completelyoutside |= endCheck < MakeMinusTolerant<ForInside>(0.);
            if(ForInside)
                completelyinside &= endCheck > MakePlusTolerant<ForInside>(0.);
        }
        else {
            completelyoutside &= endCheck < MakeMinusTolerant<ForInside>(0.);
            if(ForInside)
               completelyinside |= endCheck > MakePlusTolerant<ForInside>(0.);
        }
    }

    };


#endif /* VECGEOM_VOLUMES_WEDGE_H_ */
