/// @file UnplacedTrapezoid.h

#ifndef VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include <string>

#include "backend/Backend.h"
#include "base/PlaneShell.h"


namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedTrapezoid; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedTrapezoid );

inline namespace VECGEOM_IMPL_NAMESPACE {

typedef Vector3D<Precision> TrapCorners_t[8];

class UnplacedTrapezoid : public VUnplacedVolume, public AlignedBase {

#ifndef VECGEOM_PLANESHELL_DISABLE
  typedef PlaneShell<4, Precision> Planes;
#else
  struct TrapSidePlane {
    Precision fA,fB,fC,fD;
    // Plane equation: Ax+By+Cz+D=0, where
    // normal unit vector nvec=(A,B,C)  and offset=D is the distance from origin to plane
  };
#endif

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

#ifndef VECGEOM_PLANESHELL_DISABLE
  Planes fPlanes;
#else
  TrapSidePlane  fPlanes[4];
#endif

  Precision sideAreas[6];  // including z-planes

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
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedTrapezoid>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
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
  UnplacedTrapezoid( TrapCorners_t const corners );

  /// \brief Constructor for "default" UnplacedTrapezoid whose parameters are to be set later
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid()
    : fDz(0.f), fTheta(0.f), fPhi(0.f), fDy1(0.f), fDx1(0.f), fDx2(0.f), fTanAlpha1(0.f)
    , fDy2(0.f), fDx3(0.f), fDx4(0.f), fTanAlpha2(0.f), fTthetaCphi(0.f), fTthetaSphi(0.f)
    , fPlanes()
  {}

  /// \brief Constructor for masquerading a box (test purposes)
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(Precision xbox, Precision ybox, Precision zbox);

  /// \brief Copy constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(UnplacedTrapezoid const &other);
  /// @}

  /// assignment operator (temporary)
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid& operator=( UnplacedTrapezoid const& other );

  /// Destructor
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~UnplacedTrapezoid();

  /// Accessors
  /// @{
  // VECGEOM_CUDA_HEADER_BOTH
  // TrapParameters const& GetParameters() const { return _params; }

#ifndef VECGEOM_PLANESHELL_DISABLE
  VECGEOM_CUDA_HEADER_BOTH
  Planes const* GetPlanes() const { return &fPlanes; }
#else
  VECGEOM_CUDA_HEADER_BOTH
  TrapSidePlane const* GetPlanes() const { return fPlanes; }
#endif

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

#ifndef VECGEOM_NVCC
  // Computes capacity of the shape in [length^3]
  Precision Capacity() const { return Volume();}

  Precision SurfaceArea() const;

  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const;

  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const;

  Vector3D<Precision>  GetPointOnSurface() const;

  Vector3D<Precision> GetPointOnPlane(Vector3D<Precision> p0, Vector3D<Precision> p1,
                                      Vector3D<Precision> p2, Vector3D<Precision> p3) const;

  std::string GetEntityType() const { return "Trapezoid";}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  void GetParameterList() const {;}

  // VECGEOM_CUDA_HEADER_BOTH
  // UnplacedTrapezoid* Clone() const {
  //   return new UnplacedTrapezoid(GetDz(), GetTheta(), GetPhi(),
  //                                GetDy1(), GetDx1(), GetDx2(), GetTanAlpha1(),
  //                                GetDy2(), GetDx3(), GetDx4(), GetTanAlpha2() );
  // }

  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& StreamInfo(std::ostream &os) const;

  Vector3D<Precision> ApproxSurfaceNormal(const Vector3D<Precision>& p) const;

  /// \brief Volume
  Precision Volume() const;

  /// \brief Calculate trapezoid parameters when user provides the 8 corners
  VECGEOM_CUDA_HEADER_BOTH
  void fromCornersToParameters( TrapCorners_t const  pt);

  /// \brief Calculate the 8 corner points using pre-stored parameters
  VECGEOM_CUDA_HEADER_BOTH
  void fromParametersToCorners( TrapCorners_t  pt ) const;

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
  VECGEOM_CUDA_HEADER_BOTH
  bool MakePlanes();

  /// \brief Construct the four side planes from input corner points
  VECGEOM_CUDA_HEADER_BOTH
  bool MakePlanes( TrapCorners_t const corners );

  /// \brief Construct a side plane containing four of the trapezoid
  /// corners defining a side face
  VECGEOM_CUDA_HEADER_BOTH
  bool MakePlane( const Vector3D<Precision>& p1, const Vector3D<Precision>& p2,
                  const Vector3D<Precision>& p3, const Vector3D<Precision>& p4,
#ifndef VECGEOM_PLANESHELL_DISABLE
                  unsigned int planeIndex );
#else
                  TrapSidePlane& plane );
#endif
};

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_
