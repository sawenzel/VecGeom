/// \file Transformation3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
#include "VecGeom/base/Transformation3D.h"

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/Assert.h"

#ifdef VECGEOM_ROOT
#include "TGeoMatrix.h"
#endif

#include <sstream>
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const Transformation3D Transformation3D::kIdentity = Transformation3D();

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision phi,
                                   const Precision theta, const Precision psi)
    : fIdentity(false), fHasRotation(true), fHasTranslation(true)
{
  SetTranslation(tx, ty, tz);
  SetRotation(phi, theta, psi);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz,
                                   Transformation3D const &trot)
    : Transformation3D(trot)
{
  SetTranslation(tx, ty, tz);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision phi,
                                   const Precision theta, const Precision psi, Precision sx, Precision sy, Precision sz)
    : fIdentity(false), fHasRotation(true), fHasTranslation(true)
{
  SetTranslation(tx, ty, tz);
  SetRotation(phi, theta, psi);
  ApplyScale(sx, sy, sz);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision rxx,
                                   const Precision ryx, const Precision rzx, const Precision rxy, const Precision ryy,
                                   const Precision rzy, const Precision rxz, const Precision ryz, const Precision rzz)
    : fIdentity(false), fHasRotation(true), fHasTranslation(true)
{
  SetTranslation(tx, ty, tz);
  SetRotation(rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision rxx,
                                   const Precision ryx, const Precision rzx, const Precision rxy, const Precision ryy,
                                   const Precision rzy, const Precision rxz, const Precision ryz, const Precision rzz,
                                   const Precision sx, const Precision sy, const Precision sz)
    : fIdentity(false), fHasRotation(true), fHasTranslation(true)
{
  SetTranslation(tx, ty, tz);
  SetRotation(rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz);
  ApplyScale(sx, sy, sz);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Vector3D<Precision> &axis, bool inverse)
{
  SetTranslation(0, 0, 0);
  if (inverse)
    SetRotation(axis.Phi() * kRadToDeg - 90, -axis.Theta() * kRadToDeg, 0);
  else
    SetRotation(0, axis.Theta() * kRadToDeg, 90 - axis.Phi() * kRadToDeg);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Vector3D<double> Transformation3D::Axis() const
{
  // Determine rotation axis. Taken from clhep/src/RotationA.cc
  const double eps = 1e-15;

  double Ux = rzy_ - ryz_;
  double Uy = rxz_ - rzx_;
  double Uz = ryx_ - rxy_;
  if (std::abs(Ux) < eps && std::abs(Uy) < eps && std::abs(Uz) < eps) {

    double cosdelta = (rxx_ + ryy_ + rzz_ - 1.0) / 2.0;
    if (cosdelta > 0.0) return Vector3D<double>(0, 0, 1); // angle = 0, any axis is good

    double mxx = (rxx_ + 1) / 2;
    double myy = (ryy_ + 1) / 2;
    double mzz = (rzz_ + 1) / 2;
    double mxy = (rxy_ + ryx_) / 4;
    double mxz = (rxz_ + rzx_) / 4;
    double myz = (ryz_ + rzy_) / 4;

    double x, y, z;
    if (mxx > ryy_ && mxx > rzz_) {
      x = sqrt(mxx);
      if (rzy_ - ryz_ < 0) x = -x;
      y = mxy / x;
      z = mxz / x;
      return Vector3D<double>(x, y, z).Unit();
    } else if (myy > mzz) {
      y = sqrt(myy);
      if (rxz_ - rzx_ < 0) y = -y;
      x = mxy / y;
      z = myz / y;
      return Vector3D(x, y, z).Unit();
    } else {
      z = std::sqrt(mzz);
      if (ryx_ - rxy_ < 0) z = -z;
      x = mxz / z;
      y = myz / z;
      return Vector3D<double>(x, y, z).Unit();
    }
  } else {
    return Vector3D(Ux, Uy, Uz).Unit();
  }
}

void Transformation3D::Print(std::ostream &s) const
{
  s << "Transformation3D {{" << tx_ << "," << ty_ << "," << tz_ << "}";
  s << "{" << rxx_ << "," << ryx_ << "," << rzx_ << "," << rxy_ << "," << ryy_ << "," << rzy_ << "," << rxz_ << ","
    << ryz_ << "," << rzz_ << "}}\n";
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::Print() const
{
  printf("Transformation3D {{%.2f, %.2f, %.2f}, ", tx_, ty_, tz_);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}}", rxx_, ryx_, rzx_, rxy_, ryy_, rzy_, rxz_, ryz_,
         rzz_);
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::PrintG4() const
{
  using vecCore::math::Abs;
  using vecCore::math::Max;
  constexpr double deviationTolerance = 1.0e-05;
  printf("  Transformation: \n");

  bool UnitTr            = !fHasRotation;
  double diagDeviation   = Max(Abs(rxx_ - 1.0), Abs(ryy_ - 1.0), Abs(rzz_ - 1.0));
  double offdDeviationUL = Max(Abs(rxy_), Abs(rxz_), Abs(ryx_));
  double offdDeviationDR = Max(Abs(ryz_), Abs(rzx_), Abs(rzy_));
  double offdDeviation   = Max(offdDeviationUL, offdDeviationDR);

  if (UnitTr || Max(diagDeviation, offdDeviation) < deviationTolerance) {
    printf("    UNIT  Rotation \n");
  } else {
    printf("rx/x,y,z: %.6g %.6g %.6g\nry/x,y,z: %.6g %.6g %.6g\nrz/x,y,z: %.6g %.6g %.6g\n", rxx_, rxy_, rxz_, ryx_,
           ryy_, ryz_, rzx_, rzy_, rzz_);
  }

  printf("tr/x,y,z: %.6g %.6g %.6g\n", tx_, ty_, tz_);
}

VECCORE_ATT_HOST_DEVICE
Transformation3D &Transformation3D::Rectify()
{
  double xx = rxx_, xy = rxy_, xz = rxz_;
  double yx = ryx_, yy = ryy_, yz = ryz_;
  double zx = rzx_, zy = rzy_, zz = rzz_;
  double det = xx * yy * zz + xy * yz * zx + xz * yx * zy - xx * yz * zy - xy * yx * zz - xz * yy * zx;

  if (det <= 0) {
    printf("Transformation3D::rectify() : Attempt to rectify a Rotation with determinant <= 0\n");
    return *this;
  }
  double di = 1.0 / det;

  // xx, xy, ... are components of inverse matrix:
  double xx1 = (yy * zz - yz * zy) * di;
  double xy1 = (zy * xz - zz * xy) * di;
  double xz1 = (xy * yz - xz * yy) * di;
  double yx1 = (yz * zx - yx * zz) * di;
  double yy1 = (zz * xx - zx * xz) * di;
  double yz1 = (xz * yx - xx * yz) * di;
  double zx1 = (yx * zy - yy * zx) * di;
  double zy1 = (zx * xy - zy * xx) * di;
  double zz1 = (xx * yy - xy * yx) * di;

  // Now average with the TRANSPOSE of that:
  rxx_ = .5 * (xx + xx1);
  rxy_ = .5 * (xy + yx1);
  rxz_ = .5 * (xz + zx1);
  ryx_ = .5 * (yx + xy1);
  ryy_ = .5 * (yy + yy1);
  ryz_ = .5 * (yz + zy1);
  rzx_ = .5 * (zx + xz1);
  rzy_ = .5 * (zy + yz1);
  rzz_ = .5 * (zz + zz1);

  // Now force feed this improved rotation
  double delta    = 0;
  double cosdelta = (rxx_ + ryy_ + rzz_ - 1.0) / 2.0;
  if (cosdelta > 1.0) {
    delta = 0;
  } else if (cosdelta < -1.0) {
    delta = kPi;
  } else {
    delta = acos(cosdelta);
  }
  Vector3D<double> u = Axis();
  u                  = u.Unit(); // Because if the rotation is inexact, then the
                                 // axis() returned will not have length 1!
  Set(u, delta);
  return *this;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::Set(Vector3D<double> const &aaxis, double ddelta)
{
  double sinDelta         = sin(ddelta);
  double cosDelta         = cos(ddelta);
  double oneMinusCosDelta = 1.0 - cosDelta;

  Vector3D<double> u = aaxis.Unit();

  double uX = u[0];
  double uY = u[1];
  double uZ = u[2];

  rxx_ = oneMinusCosDelta * uX * uX + cosDelta;
  rxy_ = oneMinusCosDelta * uX * uY - sinDelta * uZ;
  rxz_ = oneMinusCosDelta * uX * uZ + sinDelta * uY;

  ryx_ = oneMinusCosDelta * uY * uX + sinDelta * uZ;
  ryy_ = oneMinusCosDelta * uY * uY + cosDelta;
  ryz_ = oneMinusCosDelta * uY * uZ - sinDelta * uX;

  rzx_ = oneMinusCosDelta * uZ * uX - sinDelta * uY;
  rzy_ = oneMinusCosDelta * uZ * uY + sinDelta * uX;
  rzz_ = oneMinusCosDelta * uZ * uZ + cosDelta;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetTranslation(const Precision tx, const Precision ty, const Precision tz)
{
  tx_ = tx;
  ty_ = ty;
  tz_ = tz;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetTranslation(Vector3D<Precision> const &vec) { SetTranslation(vec[0], vec[1], vec[2]); }

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetProperties()
{
  fHasTranslation = (fabs(tx_) > kTolerance || fabs(ty_) > kTolerance || fabs(tz_) > kTolerance) ? true : false;
  fHasRotation    = (fabs(rxx_ - 1.) > kTolerance) || (fabs(ryx_) > kTolerance) || (fabs(rzx_) > kTolerance) ||
                 (fabs(rxy_) > kTolerance) || (fabs(ryy_ - 1.) > kTolerance) || (fabs(rzy_) > kTolerance) ||
                 (fabs(rxz_) > kTolerance) || (fabs(ryz_) > kTolerance) || (fabs(rzz_ - 1.) > kTolerance);
  fIdentity = !fHasTranslation && !fHasRotation;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetRotation(const Precision phi, const Precision theta, const Precision psi)
{

  const Precision sinphi = sin(kDegToRad * phi);
  const Precision cosphi = cos(kDegToRad * phi);
  const Precision sinthe = sin(kDegToRad * theta);
  const Precision costhe = cos(kDegToRad * theta);
  const Precision sinpsi = sin(kDegToRad * psi);
  const Precision cospsi = cos(kDegToRad * psi);

  rxx_ = cospsi * cosphi - costhe * sinphi * sinpsi;
  ryx_ = -sinpsi * cosphi - costhe * sinphi * cospsi;
  rzx_ = sinthe * sinphi;
  rxy_ = cospsi * sinphi + costhe * cosphi * sinpsi;
  ryy_ = -sinpsi * sinphi + costhe * cosphi * cospsi;
  rzy_ = -sinthe * cosphi;
  rxz_ = sinpsi * sinthe;
  ryz_ = cospsi * sinthe;
  rzz_ = costhe;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetRotation(Vector3D<Precision> const &vec) { SetRotation(vec[0], vec[1], vec[2]); }

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetRotation(const Precision xx, const Precision yx, const Precision zx, const Precision xy,
                                   const Precision yy, const Precision zy, const Precision xz, const Precision yz,
                                   const Precision zz)
{

  rxx_ = xx;
  ryx_ = yx;
  rzx_ = zx;
  rxy_ = xy;
  ryy_ = yy;
  rzy_ = zy;
  rxz_ = xz;
  ryz_ = yz;
  rzz_ = zz;
}

VECCORE_ATT_HOST_DEVICE
Transformation3D &Transformation3D::RotateX(double a)
{
  double c  = cos(a);
  double s  = sin(a);
  double x1 = ryx_, y1 = ryy_, z1 = ryz_;
  ryx_ = c * x1 - s * rzx_;
  ryy_ = c * y1 - s * rzy_;
  ryz_ = c * z1 - s * rzz_;
  rzx_ = s * x1 + c * rzx_;
  rzy_ = s * y1 + c * rzy_;
  rzz_ = s * z1 + c * rzz_;

  double tx = tx_;
  double ty = c * ty_ - s * tz_;
  double tz = s * ty_ + c * tz_;
  SetTranslation(tx, ty, tz);
  SetProperties();
  return *this;
}

VECCORE_ATT_HOST_DEVICE
Transformation3D &Transformation3D::RotateY(double a)
{
  double c  = cos(a);
  double s  = sin(a);
  double x1 = rzx_, y1 = rzy_, z1 = rzz_;
  rzx_ = c * x1 - s * rxx_;
  rzy_ = c * y1 - s * rxy_;
  rzz_ = c * z1 - s * rxz_;
  rxx_ = s * x1 + c * rxx_;
  rxy_ = s * y1 + c * rxy_;
  rxz_ = s * z1 + c * rxz_;

  double tx = c * tx_ + s * tz_;
  double ty = ty_;
  double tz = -s * tx_ + c * tz_;
  SetTranslation(tx, ty, tz);
  SetProperties();
  return *this;
}

VECCORE_ATT_HOST_DEVICE
Transformation3D &Transformation3D::RotateZ(double a)
{
  double c  = cos(a);
  double s  = sin(a);
  double x1 = rxx_, y1 = rxy_, z1 = rxz_;
  rxx_ = c * x1 - s * ryx_;
  rxy_ = c * y1 - s * ryy_;
  rxz_ = c * z1 - s * ryz_;
  ryx_ = s * x1 + c * ryx_;
  ryy_ = s * y1 + c * ryy_;
  ryz_ = s * z1 + c * ryz_;

  double tx = c * tx_ - s * ty_;
  double ty = s * tx_ + c * ty_;
  double tz = tz_;
  SetTranslation(tx, ty, tz);
  SetProperties();
  return *this;
}

#ifdef VECGEOM_ROOT
// function to convert this transformation to a TGeo transformation
// mainly used for the benchmark comparisons with ROOT
TGeoMatrix *Transformation3D::ConvertToTGeoMatrix(Transformation3D const &ttd)
{
  double rotd[9];
  if (ttd.HasRotation()) {
    for (auto i = 0; i < 9; ++i)
      rotd[i] = ttd.Rotation()[i];
  }

  if (ttd.IsIdentity()) {
    return new TGeoIdentity();
  }
  if (ttd.HasTranslation() && !ttd.HasRotation()) {
    return new TGeoTranslation(ttd.Translation(0), ttd.Translation(1), ttd.Translation(2));
  }
  if (ttd.HasRotation() && !ttd.HasTranslation()) {
    TGeoRotation *tmp = new TGeoRotation();
    tmp->SetMatrix(rotd);
    return tmp;
  }
  if (ttd.HasTranslation() && ttd.HasRotation()) {
    TGeoRotation *tmp = new TGeoRotation();
    tmp->SetMatrix(rotd);
    return new TGeoCombiTrans(ttd.Translation(0), ttd.Translation(1), ttd.Translation(2), tmp);
  }
  return nullptr;
}
#endif

std::ostream &operator<<(std::ostream &os, Transformation3D const &transformation)
{
  os << "Transformation {" << transformation.Translation() << ", "
     << "(" << transformation.Rotation(0) << ", " << transformation.Rotation(1) << ", " << transformation.Rotation(2)
     << ", " << transformation.Rotation(3) << ", " << transformation.Rotation(4) << ", " << transformation.Rotation(5)
     << ", " << transformation.Rotation(6) << ", " << transformation.Rotation(7) << ", " << transformation.Rotation(8)
     << ")}"
     << "; identity(" << transformation.IsIdentity() << "); rotation(" << transformation.HasRotation() << ")";
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::Transformation3D> Transformation3D::CopyToGpu(DevicePtr<cuda::Transformation3D> const gpu_ptr) const
{

  gpu_ptr.Construct(tx_, ty_, tz_, rxx_, ryx_, rzx_, rxy_, ryy_, rzy_, rxz_, ryz_, rzz_);
  VECGEOM_DEVICE_API_CALL(GetLastError());
  return gpu_ptr;
}

DevicePtr<cuda::Transformation3D> Transformation3D::CopyToGpu() const
{

  DevicePtr<cuda::Transformation3D> gpu_ptr;
  gpu_ptr.Allocate();
  return this->CopyToGpu(gpu_ptr);
}

/**
 * Copy a large number of transformation instances to the GPU.
 * \param trafos Host instances to copy.
 * \param gpu_ptrs Device pointers to indicate where the transformations should be placed.
 * The device memory must have been allocated before copying.
 */
void Transformation3D::CopyManyToGpu(const std::vector<Transformation3D const *> &trafos,
                                     const std::vector<DevicePtr<cuda::Transformation3D>> &gpu_ptrs)
{
  VECGEOM_ASSERT(trafos.size() == gpu_ptrs.size());

  // Memory for constructor data
  // Store it as
  // tx0, tx1, tx2, ...,
  // ty0, ty1, ty2, ...,
  // ...
  // rot0_0, rot0_1, rot0_2, ...,
  // ...
  std::vector<Precision> trafoData(12 * trafos.size());

  std::size_t trafoCounter = 0;
  for (Transformation3D const *trafo : trafos) {
    for (unsigned int i = 0; i < 3; ++i)
      trafoData[trafoCounter + i * trafos.size()] = trafo->Translation(i);
    for (unsigned int i = 0; i < 9; ++i)
      trafoData[trafoCounter + (i + 3) * trafos.size()] = trafo->Rotation(i);
    ++trafoCounter;
  }

  ConstructManyOnGpu<cuda::Transformation3D>(
      trafos.size(), gpu_ptrs.data(), trafoData.data(), trafoData.data() + 1 * trafos.size(),
      trafoData.data() + 2 * trafos.size(), // translations
      trafoData.data() + 3 * trafos.size(), trafoData.data() + 4 * trafos.size(),
      trafoData.data() + 5 * trafos.size(), // rotations
      trafoData.data() + 6 * trafos.size(), trafoData.data() + 7 * trafos.size(), trafoData.data() + 8 * trafos.size(),
      trafoData.data() + 9 * trafos.size(), trafoData.data() + 10 * trafos.size(),
      trafoData.data() + 11 * trafos.size());
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::Transformation3D>::SizeOf();
template void DevicePtr<cuda::Transformation3D>::Construct(const Precision tx, const Precision ty, const Precision tz,
                                                           const Precision r0, const Precision r1, const Precision r2,
                                                           const Precision r3, const Precision r4, const Precision r5,
                                                           const Precision r6, const Precision r7,
                                                           const Precision r8) const;
template void ConstructManyOnGpu<Transformation3D>(std::size_t, DevicePtr<cuda::Transformation3D> const *,
                                                   Precision const *tx, Precision const *ty, Precision const *tz,
                                                   Precision const *r0, Precision const *r1, Precision const *r2,
                                                   Precision const *r3, Precision const *r4, Precision const *r5,
                                                   Precision const *r6, Precision const *r7, Precision const *r8);

} // namespace cxx

#endif // VECCORE_CUDA

} // namespace vecgeom
