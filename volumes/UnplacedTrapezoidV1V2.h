/// @file UnplacedTrapezoid.h

#ifndef VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

#include "backend/Backend.h"
#include "base/PlaneShell.h"

#ifndef VECGEOM_NVCC
  #if (defined(VECGEOM_VC) || defined(VECGEOM_VC_ACCELERATION))
    #include <Vc/Vc>
  #endif
#endif

namespace VECGEOM_NAMESPACE {

typedef Vector3D<Precision> TrapCorners_t[8];

struct TrapSidePlane {
    Precision fA,fB,fC,fD;    // Normal unit vector (a,b,c)  and offset (d)
    // => Ax+By+Cz+D=0
};

typedef PlaneShell<4,Precision> Planes;

class UnplacedTrapezoid : public VUnplacedVolume
#ifdef VECGEOM_VC_ACCELERATION
                        , public Vc::VectorAlignedBase
#endif
{

private:

  Precision fDz;
  Precision fTheta;
  Precision fPhi;
  Precision fDy1;
  Precision fDx1;
  Precision fDx2;
  Precision fTanAlpha1;
  Precision fDy2;
  Precision fDx3;
  Precision fDx4;
  Precision fTanAlpha2;
  Precision fTthetaCphi;
  Precision fTthetaSphi;

  TrapSidePlane  fPlanes[4];
  Planes fPlanes2;

public:

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

private:

  virtual void Print(std::ostream &os) const;

//***** Here is the trapezoid-specific code

public:
  /// \brief Constructors
  /// @{
  /// \brief General constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(Precision pDz, Precision pTheta, Precision pPhi,
               Precision pDy1, Precision pDx1, Precision pDx2, Precision pTanAlpha1,
               Precision pDy2, Precision pDx3, Precision pDx4, Precision pTanAlpha2 );

  /// \brief Fast constructor: all parameters from one array
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(Precision const* params );

  /// \brief Constructor based on 8 corner points
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid( TrapCorners_t const& corners );

  /// \brief Constructor for "default" UnplacedTrapezoid whose parameters are to be set later
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid() {}

  /// \brief Copy constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(UnplacedTrapezoid const &other);
  /// @}

  /// assignment operator (temporary)
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid& operator=( UnplacedTrapezoid const& other );

  /// Destructor
  ~UnplacedTrapezoid();

  /// Accessors
  /// @{
  // VECGEOM_CUDA_HEADER_BOTH
  // TrapParameters const& GetParameters() const { return _params; }

  VECGEOM_CUDA_HEADER_BOTH
  TrapSidePlane const* GetPlanes() const { return fPlanes; }

  VECGEOM_CUDA_HEADER_BOTH
  Planes const* GetPlanes2() const { return &fPlanes2; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDz()  const { return fDz; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return fTheta; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return fPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDy1() const { return fDy1; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx1() const { return fDx1; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx2() const { return fDx2; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha1() const { return fTanAlpha1; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDy2() const { return fDy2; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx3() const { return fDx3; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx4() const { return fDx4; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha2() const { return fTanAlpha2; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const { return fTthetaSphi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const { return fTthetaCphi; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha1() const { return atan(fTanAlpha1); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha2() const { return atan(fTanAlpha2); }

  /// @}


  /// \brief Volume
  VECGEOM_CUDA_HEADER_BOTH
  Precision Volume() const;

  /// \brief Calculate trapezoid parameters when user provides the 8 corners
  void fromCornersToParameters( TrapCorners_t const & pt);

  /// \brief Calculate the 8 corner points using pre-stored parameters
  void fromParametersToCorners( TrapCorners_t & pt ) const;

private:

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement = NULL) const;

  /// \brief Construct the four side planes from pre-stored parameters.
  ///
  /// The 4-points used to build each side plane must be co-planar, otherwise
  /// the program will stop after printing an error message.
  bool MakePlanes();

  /// \brief Construct the four side planes from input corner points
  bool MakePlanes( TrapCorners_t const & corners );

  /// \brief Construct a side plane containing four of the trapezoid
  /// corners defining a side face
  bool MakePlane( const Vector3D<Precision>& p1, const Vector3D<Precision>& p2,
                  const Vector3D<Precision>& p3, const Vector3D<Precision>& p4,
                  TrapSidePlane& plane );

  bool MakePlane2( const Vector3D<Precision>& p1, const Vector3D<Precision>& p2,
                   const Vector3D<Precision>& p3, const Vector3D<Precision>& p4,
                   unsigned int planeIndex );
};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_
