/*
 * \file GenTrapStruct.h
 * \brief Parameter/storage struct and small helper types for a Generic Trapezoid (Arb8).
 *
 * This header contains:
 *  - Low-level helpers in ::arb4helpers for lateral bilinear “arb4” faces:
 *      * Arb4Surf     : implicit/bilinear face equation and cheap safety bounds
 *      * BoundingPlane: robust plane representation used by the mesh bounder
 *      * Arb4Mesh     : two-triangle per side bounding slab per face for fast culling
 *  - GenTrapStruct<T> : a POD-like container of all precomputed data for a GenTrap
 *
 * No heavy kernels live here; kernels are in GenTrapImplementation.h and only consume
 * the data populated by GenTrapStruct::Initialize().
 *
 *  Created on: 17.07.2016
 *      Author: mgheata
 *  Revised on: 15.09.2025 andrei.gheata@cern.ch
 */

#ifndef VECGEOM_VOLUMES_GENTRAPSTRUCT_H_
#define VECGEOM_VOLUMES_GENTRAPSTRUCT_H_

#include <VecGeom/base/Global.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/base/Vector2D.h>
#ifndef VECCORE_CUDA
#include <VecGeom/volumes/TessellatedSection.h>
#endif

// ------------------------------------------------------------------
// Hybrid precompute switch:
// 1 (default): store a few cheap per-face scalars (tiny memory cost)
// 0           : compute everything on-the-fly
#ifndef GENTRAP_USE_HYBRID_COEFFS
#define GENTRAP_USE_HYBRID_COEFFS 1
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace arb4helpers {

/**
 * \brief Implicit representation for a lateral face of the GenTrap.
 *
 * For twisted faces the zero set is a quadratic in (x,y,z):
 * \code
 *   f(x,y,z) = (A x + B y + C z) z + D x + E y + F z + G
 * \endcode
 * with outward orientation chosen by construction. For planar faces the
 * quadratic part is zero (A=B=C=0) and the equation reduces to the plane
 * \( D x + E y + F z + G = 0 \).
 *
 * The struct also stores a per-face Lipschitz constant (4*k) used to build
 * conservative safety underestimates, computed from (A,B,C).
 *
 * \tparam Real_v scalar or vector floating-point type
 */
template <typename Real_v>
struct Arb4Surf {
  Real_v f4k{0.}; ///< 4×Lipschitz constant used in SafetyLipschitz()
  Real_v A{0.};   ///< Bilinear coefficient (quadratic in z): (A x + B y + C z) z
  Real_v B{0.};   ///< Bilinear coefficient (quadratic in z): (A x + B y + C z) z
  Real_v C{0.};   ///< Bilinear coefficient (quadratic in z): (A x + B y + C z) z
  Real_v D{0.};   ///< Planar term coefficient for x
  Real_v E{0.};   ///< Planar term coefficient for y
  Real_v F{0.};   ///< Planar term coefficient for z
  Real_v G{0.};   ///< Constant term

  /**
   * \brief Set all coefficients and precompute Lipschitz bound.
   *
   * The Lipschitz constant \c k is \f$|C|+\sqrt{A^2+B^2+C^2}\f$, we store 4*k to
   * avoid multiplications in SafetyLipschitz().
   */
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void Set(Real_v a, Real_v b, Real_v c, Real_v d, Real_v e, Real_v f,
                                                        Real_v g)
  {
    // Coefficients
    A = a;
    B = b;
    C = c;
    D = d;
    E = e;
    F = f;
    G = g;
    // Lipschitz constant
    Real_v k = Abs(c) + Sqrt(a * a + b * b + c * c);
    f4k      = 4. * k;
  }

  /// \brief No-op for API symmetry with vector lanes.
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void SyncLane() {}

  /// \brief Evaluate the full (twisted) implicit function \(f(x,y,z)\).
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v Evaluate(Vector3D<Real_v> const &p) const
  {
    return (A * p.x() + B * p.y() + C * p.z()) * p.z() + D * p.x() + E * p.y() + F * p.z() + G;
  }

  /// \brief Evaluate the planar part \( D x + E y + F z + G \).
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v EvaluatePlanar(Vector3D<Real_v> const &p) const
  {
    return D * p.x() + E * p.y() + F * p.z() + G;
  }

  /// \brief Gradient of the full (twisted) implicit function at \p p.
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector3D<Real_v> EvaluateGradient(Vector3D<Real_v> const &p) const
  {
    return Vector3D<Real_v>(A * p.z() + D, B * p.z() + E, A * p.x() + B * p.y() + 2. * C * p.z() + F);
  }

  /// \brief Dot product of the planar normal (D,E,F) with a vector \p v.
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v DotPlaneNormal(Vector3D<Real_v> const &v) const
  {
    return D * v.x() + E * v.y() + F * v.z();
  }

  /// \brief Return the (outward) plane normal (D,E,F). For twisted faces this is the linear part’s normal.
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector3D<Real_v> GetPlaneNormal() const { return {D, E, F}; }

  /**
   * \brief Conservative safety underestimate to this surface via Lipschitz bound.
   *
   * Computes a tight but conservative \em signed distance-like value based on
   * the function value and its gradient norm (see Geant4 MR5261 notes).
   * Result is positive when \c Evaluate(p) has outward sign.
   */
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v SafetyLipschitz(Vector3D<Real_v> const &p) const
  {
    auto gradx    = A * p.z() + D;
    auto grady    = B * p.z() + E;
    auto Cz       = C * p.z();
    auto CzF      = Cz + F;
    auto gradz    = A * p.x() + B * p.y() + Cz + CzF;
    auto fun      = gradx * p.x() + grady * p.y() + CzF * p.z() + G;
    auto grad2    = gradx * gradx + grady * grady + gradz * gradz;
    auto divisor  = Sqrt(grad2) + Sqrt(grad2 + f4k * Abs(fun));
    auto distSurf = 2. * fun / divisor;
    return distSurf;
  }

  /**
   * \brief Build the quadratic coefficients a,b,c for f(p+t d) along a ray.
   *
   * Using the implicit form:
   *   f(x,y,z) = (A x + B y + C z) z + D x + E y + F z + G
   * with p=(px,py,pz), d=(dx,dy,dz), one obtains:
   *   S1   = A*dx + B*dy + C*dz
   *   S0   = A*px + B*py + C*pz
   *   Ldir = D*dx + E*dy + F*dz
   *   Lpt  = D*px + E*py + F*pz
   *   a = S1*dz
   *   b = S1*pz + S0*dz + Ldir
   *   c = S0*pz + Lpt + G
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE void RayQuadratic(Vector3D<Real_v> const &p, Vector3D<Real_v> const &d, Real_v &a, Real_v &b,
                                         Real_v &c) const
  {
    const Real_v S1   = A * d.x() + B * d.y() + C * d.z();
    const Real_v S0   = A * p.x() + B * p.y() + C * p.z();
    const Real_v Ldir = D * d.x() + E * d.y() + F * d.z();
    const Real_v Lpt  = D * p.x() + E * p.y() + F * p.z();
    a                 = S1 * d.z();
    b                 = S1 * p.z() + S0 * d.z() + Ldir;
    c                 = S0 * p.z() + Lpt + G;
  }
};

/**
 * \brief Specialization for double with 16B-aligned packed coefficients.
 *
 * This packs coefficients in 16-byte “wide” pairs to help both host SIMD and
 * device (CUDA) vector loads (e.g. \c ld.global.v2.f64).
 */
template <>
struct alignas(16) Arb4Surf<double> {

  struct alignas(16) CoeffLane {
    wide::d2 la, bc, de, fg; // four 16B chunks
  } lane;

  /// \copydoc Arb4Surf::Set
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE void Set(double a, double b, double c, double d, double e, double f,
                                                        double g)
  {
    // Lipschitz constant
    double k = Abs(c) + Sqrt(a * a + b * b + c * c);
    lane.la.set(4. * k, a);
    lane.bc.set(b, c);
    lane.de.set(d, e);
    lane.fg.set(f, g);
  }

  /// \copydoc Arb4Surf::Evaluate
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE double Evaluate(const Vector3D<double> &p) const
  {
    // Each of these temporaries copies a 16B object → NVCC emits ld.global.v2.f64 on device
    auto la = lane.la, bc = lane.bc, de = lane.de, fg = lane.fg;
    double ax = la.y() * p.x(), by = bc.x() * p.y();
    double cz = bc.y() * p.z(), dx = de.x() * p.x();
    double ey = de.y() * p.y(), fz = fg.x() * p.z();
    return (ax + by + cz) * p.z() + dx + ey + fz + fg.y();
  }

  /// \copydoc Arb4Surf::EvaluatePlanar
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE double EvaluatePlanar(Vector3D<double> const &p) const
  {
    auto de = lane.de, fg = lane.fg;
    double dx = de.x() * p.x(), ey = de.y() * p.y(), fz = fg.x() * p.z();
    return dx + ey + fz + fg.y();
  }

  /// \copydoc Arb4Surf::EvaluateGradient
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector3D<double> EvaluateGradient(const Vector3D<double> &p) const
  {
    auto la = lane.la, bc = lane.bc, de = lane.de, fg = lane.fg;
    return {la.y() * p.z() + de.x(), // A*p.z + D
            bc.x() * p.z() + de.y(), // B*p.z + E
            la.y() * p.x() + bc.x() * p.y() + 2.0 * bc.y() * p.z() + fg.x()};
  }

  /// \copydoc Arb4Surf::DotPlaneNormal
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE double DotPlaneNormal(const Vector3D<double> &v) const
  {
    auto de = lane.de, fg = lane.fg;
    return de.x() * v.x() + de.y() * v.y() + fg.x() * v.z();
  }

  /// \copydoc Arb4Surf::GetPlaneNormal
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Vector3D<double> GetPlaneNormal() const
  {
    return {lane.de.x(), lane.de.y(), lane.fg.x()};
  }

  /// \copydoc Arb4Surf::SafetyLipschitz
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE double SafetyLipschitz(Vector3D<double> const &p) const
  {
    auto la = lane.la, bc = lane.bc, de = lane.de, fg = lane.fg;
    auto gradx    = la.y() * p.z() + de.x();
    auto grady    = bc.x() * p.z() + de.y();
    auto Cz       = bc.y() * p.z();
    auto CzF      = Cz + fg.x();
    auto gradz    = la.y() * p.x() + bc.x() * p.y() + Cz + CzF;
    auto fun      = gradx * p.x() + grady * p.y() + CzF * p.z() + fg.y();
    auto grad2    = gradx * gradx + grady * grady + gradz * gradz;
    auto divisor  = Sqrt(grad2) + Sqrt(grad2 + la.x() * Abs(fun));
    auto distSurf = 2. * fun / divisor;
    return distSurf;
  }

  /**
   * \brief Build the quadratic coefficients a,b,c for f(p+t d) along a ray (double specialization).
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE void RayQuadratic(const Vector3D<double> &p, const Vector3D<double> &d, double &a, double &b,
                                         double &c) const
  {
    // Unpack coefficients once (compiler emits ld.global.v2.f64 for the pairs)
    auto la = lane.la, bc = lane.bc, de = lane.de, fg = lane.fg;
    const double A = la.y();
    const double B = bc.x();
    const double C = bc.y();
    const double D = de.x();
    const double E = de.y();
    const double F = fg.x();
    const double G = fg.y();

    const double S1   = A * d.x() + B * d.y() + C * d.z();
    const double S0   = A * p.x() + B * p.y() + C * p.z();
    const double Ldir = D * d.x() + E * d.y() + F * d.z();
    const double Lpt  = D * p.x() + E * p.y() + F * p.z();

    a = S1 * d.z();
    b = S1 * p.z() + S0 * d.z() + Ldir;
    c = S0 * p.z() + Lpt + G;
  }
};

static_assert(alignof(Arb4Surf<double>) >= 16, "Need 16B alignment");
static_assert(sizeof(Arb4Surf<double>) % 16 == 0, "Size multiple of 16 helps arrays");

/**
 * \brief Robust plane used as bounding slab for a lateral face.
 *
 * Plane is represented as \( \mathbf{n}\cdot\mathbf{x} + d = 0 \) with \c n unit length.
 * Built from three points using a robust triangle normal.
 */
template <typename Real_v>
struct BoundingPlane {

  using Vertex_t = Vector3D<Real_v>;

  Vertex_t n;   ///< unit normal
  Real_v d{0.}; ///< plane constant so that n·x + d = 0

  BoundingPlane() = default;

  /// \brief Construct from triangle (a,b,c) with robust orientation.
  VECCORE_ATT_HOST_DEVICE
  BoundingPlane(Vertex_t const &a, Vertex_t const &b, Vertex_t const &c) { Set(a, b, c); }

  /// \brief Reset plane from triangle (a,b,c).
  VECCORE_ATT_HOST_DEVICE
  void Set(Vertex_t const &a, Vertex_t const &b, Vertex_t const &c) { TriangleUnitNormalRobust(a, b, c, n, d); }

  /// \brief Signed distance from \p point (positive on the normal side).
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v SignedDistance(const Vertex_t &point) const
  {
    return n.Dot(point) + d; // because plane is n·x + d = 0
  }
};

/**
 * \brief Two-triangle bounding slab per side for a lateral (twisted/planar) face.
 *
 * Each Arb4 face is conservatively bounded by two oriented triangles. We keep
 * the “inbound/outbound” half-spaces to enable very cheap “may hit” culls and
 * fast signed safeties.
 */
template <typename Real_v>
struct Arb4Mesh {
  using Vertex_t    = Vector3D<Real_v>;
  using Direction_t = Vertex_t;

  BoundingPlane<Real_v> fBplanes[4]; ///< Two outbound + two inbound oriented planes

  Arb4Mesh() = default;

  /// \brief Build the bounding planes from the four face corners (bottom v1,v2 and top v3,v4).
  VECCORE_ATT_HOST_DEVICE
  Arb4Mesh(Vertex_t const &v1, Vertex_t const &v2, Vertex_t const &v3, Vertex_t const &v4) { Set(v1, v2, v3, v4); }

  /**
   * \brief Set planes from corners; orientation depends on whether v4 is behind (v1,v3)×(v1,v2).
   *
   * We build: 2 “outbound” planes (fBplanes[0..1]) and 2 “inbound” planes (fBplanes[2..3]).
   * The Inside/MaysHit helpers pick the appropriate pair based on whether the start is inside the solid.
   */
  VECCORE_ATT_HOST_DEVICE
  void Set(Vertex_t const &v1, Vertex_t const &v2, Vertex_t const &v3, Vertex_t const &v4)
  {
    // Check if P4 is behind (P1,P3) X (P1, P2)
    bool behind = ((v3 - v1).Cross(v2 - v1)).Dot(v4 - v1) < 0;
    if (behind) {
      // outbound triangles
      fBplanes[0].Set(v1, v3, v2);
      fBplanes[1].Set(v2, v3, v4);
      // inbound triangles
      fBplanes[2].Set(v1, v2, v4);
      fBplanes[3].Set(v1, v4, v3);
    } else {
      // outbound triangles
      fBplanes[0].Set(v1, v4, v2);
      fBplanes[1].Set(v1, v3, v4);
      // inbound triangles
      fBplanes[2].Set(v1, v2, v3);
      fBplanes[3].Set(v2, v4, v3);
    }
  }

  /// \brief True if \p point is inside all four planes.
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE bool Inside(Vertex_t const &point) const
  {
    for (auto i = 0; i < 4; ++i) {
      if (fBplanes[i].SignedDistance(point) > Real_v(0.)) return false;
    }
    return true;
  }

  /**
   * \brief Optimized inside check knowing whether the point is inside the \em solid.
   *
   * When \p point_inside is known, only two planes (either inbound or outbound) matter.
   */
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE bool Inside(Vertex_t const &point, bool point_inside) const
  {
    int i = 2 * int(point_inside);
    return (fBplanes[i].SignedDistance(point) < Real_v(0.)) && (fBplanes[i + 1].SignedDistance(point) < Real_v(0.));
  }

  /**
   * \brief Cheap necessary test for a potential hit along direction \p dir.
   *
   * If the point is outside the mesh, the ray can only hit if it points
   * against at least one of the relevant plane normals.
   */
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE bool MayHit(Vertex_t const &point, Direction_t const &dir,
                                                           bool point_inside) const
  {
    // If point is inside the mesh, the direction may always hit
    if (Inside(point, point_inside)) return true;
    // Otherwise: must be opposite to at least one normal
    int i = 2 * int(point_inside);
    return (dir.Dot(fBplanes[i].n) < Real_v(0.)) || (dir.Dot(fBplanes[i + 1].n) < Real_v(0.));
  }

  /**
   * \brief Conservative (unsigned) safety to the bounding planes.
   *
   * If the point is outside the two relevant planes, return the minimum absolute
   * distance to either plane, otherwise return 0.
   */
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v Safety(Vertex_t const &point, bool inside) const
  {
    if (Inside(point)) return Real_v(0.);
    int i     = 2 * int(inside);
    Real_v s0 = Abs(fBplanes[i].SignedDistance(point));
    Real_v s1 = Abs(fBplanes[i + 1].SignedDistance(point));
    return Min(s0, s1);
  }

  /**
   * \brief Very cheap signed safety: positive outside the slab, 0 inside, negative inside-solid mode.
   *
   * Useful for coarse pruning; not accurate vs. the true triangular patch.
   */
  VECCORE_ATT_HOST_DEVICE VECGEOM_FORCE_INLINE Real_v FastSignedSafety(Vertex_t const &point, bool inside) const
  {
    int i         = 2 * int(inside);
    Real_v safety = Max(fBplanes[i].SignedDistance(point), fBplanes[i + 1].SignedDistance(point));
    safety        = Max(Real_v(0.), safety); // truncate to 0 when inside the two planes
    safety *= Real_v(1 - i);                 // outside: +1, inside: -1  (i is 0 or 2)
    return safety;
  }
};
} // namespace arb4helpers

/**
 * \brief POD-like storage for a Generalized Trapezoid (Arb8) geometry.
 *
 * This struct owns only precomputed, immutable geometry data (vertices, per-face
 * coefficients, cheap bounders, etc.). Navigation kernels are implemented in
 * GenTrapImplementation and read from this struct.
 *
 * \tparam T floating precision
 */
template <typename T = double>
struct alignas(16) GenTrapStruct {
  using Vertex_t = Vector3D<T>;
  using PointXY  = Vector2D<T>;
  using UChar_t  = unsigned char;

  Vertex_t fBBdimensions;            ///< Half-lengths of the AABB (x,y,z)
  Vertex_t fBBorigin;                ///< Center of the AABB in local coordinates
  T fDz{0.};                         ///< Half-height along Z (top/bottom at ±fDz)
  T fDz2{0.};                        ///< Precomputed 1/(2*fDz) used for fast v(z) mapping
  Vertex_t fVertices[8];             ///< The eight corner points: 0..3 bottom (z=-Dz), 4..7 top (z=+Dz)
  Vector2D<T> fT[4];                 ///< Per-face generator slopes in XY (bottom->top) scaled by 1/(2*fDz)
  Vector2D<T> fMidXY[4];             ///< Midpoint at z=0 for each generator: v[i].XY() + fT[i]*fDz
  arb4helpers::Arb4Surf<T> fSurf[4]; ///< Twisted/planar implicit coefficients for each lateral face
  arb4helpers::Arb4Mesh<T> fMesh[4]; ///< Two-triangle bounding slabs per lateral face side
  UChar_t fFlags{0};                 ///< Bitfield: [0..3]=twisted flags per face, [4..7]=degenerated flags

#if GENTRAP_USE_HYBRID_COEFFS
  // --- Hybrid precompute (tiny storage, reduces ALU) ---------------------
  // For each lateral face i (with j=(i+1)&3), we cache:
  //   fNext[i]  = j
  //   fTT[i]    = T[i]   × T[j]
  //   fMM[i]    = Mid[i] × Mid[j]
  //   fMT[i]    = Mid[i] × T[j]  -  Mid[j] × T[i]
  int fNext[4]; ///< next index j = (i+1)%4 for face i
  T fTT[4];     ///< crossZ(T[i],   T[j])
  T fMM[4];     ///< crossZ(Mid[i], Mid[j])
  T fMT[4];     ///< crossZ(Mid[i], T[j]) - crossZ(Mid[j], T[i])

  // Extended caches (vectors) used by twisted intersection, XYZ->UV, and UNormal:
  Vector2D<T> fDmid[4]; ///< Mid[j] - Mid[i]
  Vector2D<T> fDT[4];   ///< T[j]   - T[i]
  Vector3D<T> fE0[4];   ///< C - A  (bottom edge, z=0)
  Vector3D<T> fE1[4];   ///< D - B  (top edge,    z=0)
  Vector3D<T> fG0[4];   ///< B - A  (generator at i,    z=+2Dz)
  Vector3D<T> fG1[4];   ///< D - C  (generator at j,    z=+2Dz)
#endif

#ifndef VECCORE_CUDA
  TessellatedSection<T> *fTslHelper = nullptr; ///< SIMD helper using tessellated clusters for the planar case
#endif

  GenTrapStruct() = default;

  /**
   * \brief Construct from XY vertex arrays and half-height.
   *
   * \param verticesx 8 X-coordinates (first 4 bottom, next 4 top)
   * \param verticesy 8 Y-coordinates (first 4 bottom, next 4 top)
   * \param halfzheight Half height in Z (fDz)
   */
  VECCORE_ATT_HOST_DEVICE
  GenTrapStruct(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
  {
    // Constructor
    Initialize(verticesx, verticesy, halfzheight);
  }

  /**
   * \brief Initialize all derived data from input vertices and half-height.
   *
   * Performs:
   *  - Vertex placement at z=±fDz
   *  - Degeneracy flags per lateral face
   *  - Shear vectors fT and midpoints fMidXY
   *  - Orientation fix to ensure clockwise order in XY
   *  - Robust validation: no crossing opposite segments; convex quads
   *  - Per-face classification: planar vs twisted; plane or twist coefficients
   *  - Optional TessellatedSection build for pure planar case (host only)
   *  - AABB computation
   *
   * \return true on success; validation failures are guarded by VECGEOM_VALIDATE
   */
  VECCORE_ATT_HOST_DEVICE
  bool Initialize(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
  {
    // Initialization based on vertices and half length
    fDz  = halfzheight;
    fDz2 = 1. / (2. * fDz);

    // Set vertices in Vector3D form
    for (int i = 0; i < 4; ++i) {
      fVertices[i].Set(verticesx[i], verticesy[i], -fDz);
      fVertices[i + 4].Set(verticesx[i + 4], verticesy[i + 4], fDz);
    }

    // Compute degenerated faces, generator connections and midpoints
    for (int i = 0; i < 4; ++i) {
      int j          = (i + 1) % 4;
      const auto &p1 = fVertices[i];
      const auto &p2 = fVertices[j];
      const auto &p3 = fVertices[i + 4];
      const auto &p4 = fVertices[j + 4];
      auto lbot      = (p2 - p1).Mag2();
      auto ltop      = (p4 - p3).Mag2();
      SetDegenerated(i, Max(lbot, ltop) < kToleranceSquared);
      fT[i]     = fDz2 * (fVertices[i + 4].XY() - fVertices[i].XY());
      fMidXY[i] = fVertices[i].XY() + fT[i] * fDz;
    }

#if GENTRAP_USE_HYBRID_COEFFS
    // Precompute light, read-only per-face scalars used in twisted intersection
    for (int i = 0; i < 4; ++i) {
      int j    = (i + 1) % 4;
      fNext[i] = j;
      // cross products in XY (z-component of 3D cross)
      const auto &Ti = fT[i];
      const auto &Tj = fT[j];
      const auto &Mi = fMidXY[i];
      const auto &Mj = fMidXY[j];
      fTT[i]         = Ti.CrossZ(Tj);
      fMM[i]         = Mi.CrossZ(Mj);
      fMT[i]         = Mi.CrossZ(Tj) - Mj.CrossZ(Ti);

      // Extended vector caches
      fDmid[i] = Mj - Mi;
      fDT[i]   = Tj - Ti;

      // Face vertices: A=i, C=j on bottom; B=i+4, D=j+4 on top
      const auto &A = fVertices[i];
      const auto &C = fVertices[j];
      const auto &B = fVertices[i + 4];
      const auto &D = fVertices[j + 4];
      fE0[i]        = C - A; // z=0
      fE1[i]        = D - B; // z=0
      fG0[i]        = B - A; // z=+2Dz
      fG1[i]        = D - C; // z=+2Dz
    }
#endif

    // Make sure vertices are defined clockwise on both z-planes
    Precision sum1 = 0.;
    Precision sum2 = 0.;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      sum1 += fVertices[i].CrossZ(fVertices[j]);
      sum2 += fVertices[i + 4].CrossZ(fVertices[j + 4]);
    }

    // We should generate an exception here
    if (sum1 * sum2 < -kTolerance) Print();
    VECGEOM_VALIDATE(sum1 * sum2 > -kTolerance, << "Unplaced generic trap defined with opposite clockwise in XY");

    // Revert sequence of vertices to have them clockwise if needed (bottom/top simultaneously)
    if (sum1 > kTolerance) {
      printf("INFO: Reverting to clockwise vertices of GenTrap shape:\n");
      Print();
      auto swap = [](auto &a, auto &b) {
        auto t = a;
        a      = b;
        b      = t;
      };
      swap(fVertices[1], fVertices[3]);
      swap(fVertices[5], fVertices[7]);
    }

    // Check that opposite segments are not crossing -> fatal exception
    bool xing0123 = SegmentsCrossing(fVertices[0], fVertices[1], fVertices[3], fVertices[2]);
    bool xing0312 = SegmentsCrossing(fVertices[1], fVertices[2], fVertices[0], fVertices[3]);
    bool xing4567 = SegmentsCrossing(fVertices[4], fVertices[5], fVertices[7], fVertices[6]);
    bool xing4756 = SegmentsCrossing(fVertices[5], fVertices[6], fVertices[4], fVertices[7]);
    if (xing0123 || xing0312 || xing4567 || xing4756) Print();
    VECGEOM_VALIDATE(!xing0123, << "Unplaced generic trap defined with crossing opposite segments (01) (23)");
    VECGEOM_VALIDATE(!xing0312, << "Unplaced generic trap defined with crossing opposite segments (03) (12)");
    VECGEOM_VALIDATE(!xing4567, << "Unplaced generic trap defined with crossing opposite segments (45) (67)");
    VECGEOM_VALIDATE(!xing4756, << "Unplaced generic trap defined with crossing opposite segments (47) (56)");

    // Check that top and bottom quadrilaterals are convex
    bool convexquads = ComputeIsConvexQuadrilaterals();
    if (!convexquads) Print();
    VECGEOM_VALIDATE(convexquads, << "Unplaced generic trap defined with top/bottom quadrilaterals not convex");

    // Mark twisted faces and precompute per-face coefficients/mesh
    ComputeTwistedFaces();

    // fSurfaceShell.Initialize(fVertices, fDz);

#ifndef VECCORE_CUDA
    //  Create the tessellated helper if the faces are planar (host-side acceleration)
    if (IsPlanar()) {
      fTslHelper = new TessellatedSection<T>(4, -fDz, fDz);
      fTslHelper->AddQuadrilateralFacet(fVertices[0], fVertices[4], fVertices[5], fVertices[1]);
      fTslHelper->AddQuadrilateralFacet(fVertices[1], fVertices[5], fVertices[6], fVertices[2]);
      fTslHelper->AddQuadrilateralFacet(fVertices[2], fVertices[6], fVertices[7], fVertices[3]);
      fTslHelper->AddQuadrilateralFacet(fVertices[3], fVertices[7], fVertices[4], fVertices[0]);
    }
#endif
    ComputeBoundingBox();
    return true;
  }

  /// \brief True if all four lateral faces are planar.
  VECCORE_ATT_HOST_DEVICE
  bool IsPlanar() const { return (fFlags & 0x0F) == 0; }

  /// \brief True if lateral face \p i is twisted (non-planar).
  VECCORE_ATT_HOST_DEVICE
  bool IsTwisted(int i) const { return (fFlags >> i) & 1u; }

  /// \brief Set twisted flag for face \p i.
  VECCORE_ATT_HOST_DEVICE
  void SetTwisted(int i, bool flag)
  {
    UChar_t mask = 1u << i;
    if (flag)
      fFlags |= mask;
    else
      fFlags &= ~mask;
  }

  /// \brief True if lateral face \p i is degenerate (collapsed segment on top or bottom).
  VECCORE_ATT_HOST_DEVICE
  bool IsDegenerated(int i) const { return (fFlags >> (i + 4)) & 1u; }

  /// \brief Set degenerate flag for face \p i.
  VECCORE_ATT_HOST_DEVICE
  void SetDegenerated(int i, bool flag)
  {
    UChar_t mask = 1u << (i + 4);
    if (flag)
      fFlags |= mask;
    else
      fFlags &= ~mask;
  }

  /// \brief Compute AABB origin and half-dimensions.
  VECCORE_ATT_HOST_DEVICE
  void ComputeBoundingBox()
  {
    // Computes bounding box parameters
    Vertex_t aMin, aMax;
    Extent(aMin, aMax);
    fBBorigin     = 0.5 * (aMin + aMax);
    fBBdimensions = 0.5 * (aMax - aMin);
  }

  /**
   * \brief Compute axis-aligned extent (min/max) of the solid including z=±fDz.
   *
   * \param[out] aMin minimum x/y/z
   * \param[out] aMax maximum x/y/z
   */
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vertex_t &aMin, Vertex_t &aMax) const
  {
    // Returns the full 3D cartesian extent of the solid.
    aMin = aMax = fVertices[0];
    aMin[2]     = -fDz;
    aMax[2]     = fDz;
    for (int i = 0; i < 4; ++i) {
      // lower -fDz vertices
      if (aMin[0] > fVertices[i].x()) aMin[0] = fVertices[i].x();
      if (aMax[0] < fVertices[i].x()) aMax[0] = fVertices[i].x();
      if (aMin[1] > fVertices[i].y()) aMin[1] = fVertices[i].y();
      if (aMax[1] < fVertices[i].y()) aMax[1] = fVertices[i].y();
      // upper fDz vertices
      if (aMin[0] > fVertices[i + 4].x()) aMin[0] = fVertices[i + 4].x();
      if (aMax[0] < fVertices[i + 4].x()) aMax[0] = fVertices[i + 4].x();
      if (aMin[1] > fVertices[i + 4].y()) aMin[1] = fVertices[i + 4].y();
      if (aMax[1] < fVertices[i + 4].y()) aMax[1] = fVertices[i + 4].y();
    }
  }

  /**
   * \brief Validate that the bottom and top quadrilaterals are convex.
   *
   * Assumes clockwise ordering in XY. Returns false if any consecutive edge
   * pairs produce a positive z-cross (i.e. non-convex or reversed).
   */
  VECCORE_ATT_HOST_DEVICE
  bool ComputeIsConvexQuadrilaterals()
  {
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      int k = (i + 2) % 4;
      // Bottom face
      auto vij     = fVertices[j] - fVertices[i];
      auto vjk     = fVertices[k] - fVertices[j];
      auto crossij = vij.Cross(vjk).z();
      if (crossij > kTolerance) return false;
      // Top face
      vij     = fVertices[j + 4] - fVertices[i + 4];
      vjk     = fVertices[k + 4] - fVertices[j + 4];
      crossij = vij.Cross(vjk).z();
      if (crossij > kTolerance) return false;
    }
    return true;
  }

  /**
   * \brief Analyze each lateral face: tag twisted/planar, precompute coefficients and mesh bounders.
   *
   * For twisted faces we compute the implicit coefficients (A..G) and normalize
   * them by the magnitude of the linear normal part (D,E,F) to maintain numeric stability.
   * For planar faces we compute the outward plane normal and its offset G.
   *
   * \return number of twisted faces
   */
  VECCORE_ATT_HOST_DEVICE
  int ComputeTwistedFaces()
  {
    // Check if the trapezoid is twisted. A lateral face is twisted if the top and
    // bottom segments are not parallel (cross product not null)

    int ntwisted = 0;

    for (int i = 0; i < 4; ++i) {
      auto j         = (i + 1) % 4;
      auto const &p1 = fVertices[i];
      auto const &p2 = fVertices[j];
      auto const &p3 = fVertices[i + 4];
      auto const &p4 = fVertices[j + 4];
      auto v12       = p2 - p1;
      auto v34       = p4 - p3;
      auto lbot      = v12.Length();
      auto ltop      = v34.Length();
      auto zcross    = v12.CrossZ(v34);
      auto eps       = kTolerance * Max(lbot, ltop);
      bool planar    = (Abs(zcross) < eps) || (Min(lbot, ltop) < kTolerance);
      SetTwisted(i, !planar);
      ntwisted += !planar;
      if (!planar) {
        // Build bounding slab for culling
        fMesh[i].Set(fVertices[i], fVertices[j], fVertices[i + 4], fVertices[j + 4]);

        // Implicit coefficients for bilinear patch (outward orientation)
        T a, b, c, d, e, f, g;
        a = -2. * fDz * (v34.y() - v12.y());
        b = 2. * fDz * (v34.x() - v12.x());
        c = -(p4 - p2).CrossZ(p3 - p1);
        d = -2. * fDz * fDz * (v34.y() + v12.y());
        e = 2. * fDz * fDz * (v34.x() + v12.x());
        f = 2. * fDz * (p3.CrossZ(p4) - p1.CrossZ(p2));
        g = -fDz * fDz * ((p4 + p2).CrossZ(p1 + p3));

        // Normalize by the magnitude of (d,e,f) to reduce scale sensitivity and store planar faces normalized
        // coefficients
        auto magnitude = Vertex_t(d, e, f).Mag();
        VECGEOM_VALIDATE(magnitude > kToleranceDist<T>, << "Wrong twist parameters");
        a /= magnitude;
        b /= magnitude;
        c /= magnitude;
        d /= magnitude;
        e /= magnitude;
        f /= magnitude;
        g /= magnitude;
        fSurf[i].Set(a, b, c, d, e, f, g);
      } else {
        // Compute normal
        auto normal = Vertex_t::Cross(p3 - p2, p4 - p1);
        if (normal.Mag2() < kTolerance) {
          normal.Set(0., 0., 1.); // No surface, just a line
        }
        normal.Normalize();

        T d, e, f, g;
        d = normal.x();
        e = normal.y();
        f = normal.z();
        g = -normal.Dot((p1 + p2 + p3 + p4) / 4.); // plane through the face centroid
        fSurf[i].Set(0., 0., 0., d, e, f, g);
      }
    }
    return ntwisted;
  }

  /**
   * \brief Convenience function compatible with Geant4 “twist” reporting.
   *
   * Returns a signed angle between bottom edge (i->i+1) and top edge ((i+4)->(i+5))
   * with sign taken from the z-cross product.
   */
  VECCORE_ATT_HOST_DEVICE
  Precision GetTwist(int i) const
  {
    if (!IsTwisted(i)) return 0.;
    auto j      = (i + 1) % 4;
    auto AC     = fVertices[j] - fVertices[i];
    auto BD     = fVertices[j + 4] - fVertices[i + 4];
    auto lbot   = AC.Length();
    auto ltop   = BD.Length();
    auto zcross = AC.CrossZ(BD);
    auto angle  = vecCore::math::ACos(AC.Dot(BD) / (lbot * ltop));
    return CopySign(angle, zcross);
  }

  /// \brief Debug print of geometry and per-face classification.
  VECCORE_ATT_HOST_DEVICE
  void Print() const
  {
    printf("UnplacedGenTrap: { halfZ: %f mm,  planar: %s }\n", fDz, (IsPlanar() ? "true" : "false"));
    printf("bottom:");
    for (int i = 0; i < 8; ++i) {
      printf(" %d:{%f, %f}", i, fVertices[i].x(), fVertices[i].y());
      if (i == 3) printf("\ntop:   ");
    }
    printf("\nlateral:");
    for (int i = 0; i < 4; ++i) {
      auto j = (i + 1) % 4;
      printf(" %d%d%d%d:", i, j, j + 4, i + 4);
      if (IsDegenerated(i))
        printf(" degenerated");
      else {
        if (IsTwisted(i))
          printf(" twisted");
        else
          printf(" planar");
      }
    }
    printf("\n");
  }

  /**
   * \brief Check whether two XY segments (p1,p2) and (q1,q2) properly cross.
   *
   * Colinear/parallel/degenerate pairs return false. Crossing is accepted only if
   * both parametric coordinates t and u lie in (kTolerance, 1-kTolerance).
   */
  VECCORE_ATT_HOST_DEVICE
  bool SegmentsCrossing(Vertex_t const &p1, Vertex_t const &p2, Vertex_t const &q1, Vertex_t const &q2) const
  {
    auto r         = p2 - p1;
    auto s         = q2 - q1;
    auto r_cross_s = r.CrossZ(s);
    if (r_cross_s < kTolerance) // parallel, colinear or degenerated - ignore crossing
      return false;
    // The segments are crossing if:
    //   t = ((q-p) × s) / (r × s)   and   u = ((q-p) × r) / (r × s)
    // fall into (kTolerance, 1-kTolerance)
    auto t = (q1 - p1).CrossZ(s) / r_cross_s;
    if (t < kTolerance || t > 1. - kTolerance) return false;
    auto u = (q1 - p1).CrossZ(r) / r_cross_s;
    if (u < kTolerance || u > 1. - kTolerance) return false;
    return true;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
