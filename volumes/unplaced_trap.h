/**
 * @file unplaced_trap.h
 * @author Guilherme Lima (Guilherme.Lima@cern.ch)
 *
 * Class description:
 *
 *   Trap is a general trapezoid: The faces perpendicular to the
 *   z planes are trapezia, and their centres are not necessarily on
 *   a line parallel to the z axis.
 *
 *   Note that of the 11 parameters described below, only 9 are really
 *   independent - a check for planarity is made in the calculation of the
 *   equation for each plane. If the planes are not parallel, a call to
 *   G4Exception is made.
 *
 *     _fDz     Half-length along the z-axis
 *     _pTheta  Polar angle of the line joining the centres of the faces
 *              at -/+pDz
 *     _pPhi    Azimuthal angle of the line joing the centre of the face at
 *              -pDz to the centre of the face at +pDz
 *
 *              Implementation note: for performance purposes, parameters pTheta,pPhi
 *               are replaced by _fTthetaCphi, _pTthetaSphi
 *
 *     _fDy1    Half-length along y of the face at -pDz
 *     _fDx1    Half-length along x of the side at y=-pDy1 of the face at -pDz
 *     _fDx2    Half-length along x of the side at y=+pDy1 of the face at -pDz
 *     _fTalpha1 Tangent of angle with respect to the y axis from the centre of the side
 *              at y=-pDy1 to the centre at y=+pDy1 of the face at -pDz
 *
 *     _fDy2    Half-length along y of the face at +pDz
 *     _fDx3    Half-length along x of the side at y=-pDy2 of the face at +pDz
 *     _fDx4    Half-length along x of the side at y=+pDy2 of the face at +pDz
 *     _fTalpha2 Tangent of angle with respect to the y axis from the centre of the side
 *              at y=-pDy2 to the centre at y=+pDy2 of the face at +pDz
 *
 * 140407 G.Lima - based on equivalent box code
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDTRAP_H_
#define VECGEOM_VOLUMES_UNPLACEDTRAP_H_

#include "volumes/unplaced_volume.h"
#include "base/vector3d.h"
#include "base/parameters.h"

namespace VECGEOM_NAMESPACE {

typedef Vector3D<Precision> TrapCorners_t[8];
typedef Parameters<11> TrapParameters;

struct TrapSidePlane {
	Precision _a,_b,_c,_d;    // Normal unit vector (a,b,c)  and offset (d)
	// => Ax+By+Cz+D=0
};

class UnplacedTrap : public VUnplacedVolume {

private: // member data
	TrapParameters _params;
	TrapSidePlane  _fPlanes[4];

public: // member functions

	/// \brief Constructors
	/// @{
	/// \briefGeneral constructor
	UnplacedTrap(Precision pDz, Precision pTheta, Precision pPhi,
				 Precision pDy1, Precision pDx1, Precision pDx2, Precision pAlpha1,
				 Precision pDy2, Precision pDx3, Precision pDx4, Precision pAlpha2 );

	/// \brief Fast constructor: all parameters from one array
	UnplacedTrap(Precision const* params );

	/// \brief Faster constructor
	UnplacedTrap( TrapParameters const& params );

	/// \brief Constructor based on 8 corner points
	UnplacedTrap( TrapCorners_t const& corners );
 
	/// \brief Constructor for "default" UnplacedTrap whose parameters are to be set later
	UnplacedTrap() {}

	/// \brief Copy constructor
	VECGEOM_CUDA_HEADER_BOTH
    UnplacedTrap(UnplacedTrap const &other);    
	/// @}

	/// \brief Total size of object in memory
	virtual int memory_size() const { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
	virtual VUnplacedVolume* CopyToGpu() const;
	virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

	VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    TrapParameters const& parameters() const { return _params; }

    // Accessors
	// @{
    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDz()  const { return _params.par(0); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDy1() const { return _params.par(1); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDx1() const { return _params.par(2); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDx2() const { return _params.par(3); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getTalpha1()    const { return _params.par(4); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDy2() const { return _params.par(5); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDx3() const { return _params.par(6); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getDx4() const { return _params.par(7); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getTalpha2()    const { return _params.par(8); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getTthetaSphi()    const { return _params.par(9); }

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision getTthetaCphi()    const { return _params.par(10); }
	// @}

    /// \brief Volume of the volume :-)
    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	Precision volume() const;

    VECGEOM_CUDA_HEADER_BOTH
      virtual void Print() const;

#ifndef VECGEOM_NVCC

    template <TranslationCode trans_code, RotationCode rot_code>
    static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                                 TransformationMatrix const *const matrix,
                                 VPlacedVolume *const placement = NULL);

    static VPlacedVolume* CreateSpecializedVolume(
        LogicalVolume const *const volume,
        TransformationMatrix const *const matrix,
        const TranslationCode trans_code, const RotationCode rot_code,
        VPlacedVolume *const placement = NULL);

#else

    template <TranslationCode trans_code, RotationCode rot_code>
    __device__
    static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               TransformationMatrix const *const matrix,
                               const int id,
                               VPlacedVolume *const placement = NULL);

    __device__
    static VPlacedVolume* CreateSpecializedVolume(
       LogicalVolume const *const volume,
       TransformationMatrix const *const matrix,
       const TranslationCode trans_code, const RotationCode rot_code,
       const int id, VPlacedVolume *const placement = NULL);

#endif

	/// \brief Calculate trapezoid parameters when user provides the 8 corners
    void fromCornersToParameters( TrapCorners_t const & pt);

    /// \brief Calcuate the 8 corner points using pre-stored parameters
    void fromParametersToCorners( TrapCorners_t & pt ) const;

private:

    virtual VPlacedVolume*
      SpecializedVolume(LogicalVolume const *const volume,
			TransformationMatrix const *const matrix,
			const TranslationCode trans_code, const RotationCode rot_code) const;

    virtual void Print(std::ostream &os) const {
		os << "Trap {" << getDz()
		   <<' '<< getDy1()
		   <<' '<< getDx1()
		   <<' '<< getDx2()
		   <<' '<< getTalpha1()
		   <<' '<< getDy2()
		   <<' '<< getDx3()
		   <<' '<< getDx4()
		   <<' '<< getTalpha2()
		   <<' '<< getTthetaCphi()
		   <<' '<< getTthetaSphi() << "}\n";
    }

    bool MakePlanes();
    bool MakePlanes( TrapCorners_t const & corners );

    bool MakePlane( const Vector3D<Precision>& p1, const Vector3D<Precision>& p2,
		    const Vector3D<Precision>& p3, const Vector3D<Precision>& p4,
		    TrapSidePlane& plane );
};

//*** FUNCTION DEFINITIONS

UnplacedTrap::UnplacedTrap(TrapParameters const& params )
	: _params(params)
{
	this->MakePlanes();
}

UnplacedTrap::UnplacedTrap(Precision const* params ) {
  for(auto i=0; i<11; ++i)
    _params[i] = params[i];

  this->MakePlanes();
} 
 
VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrap::UnplacedTrap(UnplacedTrap const &other) {

  for(auto i=0; i<4; ++i)  this->_fPlanes[i] = other._fPlanes[i];
  for(auto i=0; i<11; ++i) this->_params[i]  = other._params[i];
}

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTRAP_H_
