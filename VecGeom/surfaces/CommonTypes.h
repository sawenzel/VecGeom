#ifndef VECGEOM_SURFACE_COMMONTYPES_H
#define VECGEOM_SURFACE_COMMONTYPES_H

namespace vgbrep {

/// @brief Vector in 2D plane.
/// @tparam Real_t Floating point type.
template <typename Real_t>
struct Vec2D {
  Real_t components[2]{0};

  Vec2D() = default;

  /// @brief Constructor from potentially different type
  /// @tparam Real_s Type to construct from
  /// @param x1 First component
  /// @param x2 Second component
  template <typename Real_s>
  Vec2D(Real_s x1, Real_s x2)
  {
    components[0] = Real_s(x1);
    components[1] = Real_s(x2);
  }

  /// @brief Element accessors
  /// @param i Element index
  /// @return Element value
  Real_t operator[](int i) const { return Real_t(components[i]); }
  Real_t &operator[](int i) { return components[i]; }

  Vec2D<Real_t> operator-(const Vec2D<Real_t> &other) const
  {
    return Vec2D<Real_t>(components[0] - other.components[0], components[1] - other.components[1]);
  }

  /// @brief Vector setter
  /// @tparam Real_s Type to set from
  /// @param x1 First component
  /// @param x2 Second component
  template <typename Real_s>
  void Set(Real_s x1, Real_s x2)
  {
    components[0] = Real_t(x1);
    components[1] = Real_t(x2);
  }

  /// @brief Calculates the z-component of cross product with another 2D vector.
  /// @tparam Real_s Type stored by the other vector
  /// @param other Other vector
  /// @return Z-component of the cross product vector
  template <typename Real_s>
  Real_t CrossZ(Vec2D<Real_s> other) const
  {
    return components[0] * Real_t(other[1]) - components[1] * Real_t(other[0]);
  };

  /// @brief Calculates the dot product with another 2D vector.
  /// @tparam Real_s Type stored by the other vector
  /// @param other Other vector
  /// @return Dot product of the two vectors
  template <typename Real_s>
  Real_t Dot(Vec2D<Real_s> other) const
  {
    return components[0] * Real_t(other[1]) + components[1] * Real_t(other[1]);
  };
};

// Aliases for different usages of Vec2D.
template <typename Real_t>
using Range = Vec2D<Real_t>;

template <typename Real_t>
using AngleVector = Vec2D<Real_t>;

template <typename Real_t>
using Point2D = Vec2D<Real_t>;

/// @brief Data for cylindrical and spherical surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t, typename Real_s = Real_t>
struct CylData {
  Real_t radius{0}; ///< Cylinder radius. Stored negative if flipped.

  CylData() = default;
  CylData(Real_s rad, bool flip = false) : radius(flip ? -rad : rad) {}

  Real_s Radius() const { return std::abs(Real_s(radius)); }
  bool IsFlipped() const { return radius < 0; }
};

template <typename Real_t, typename Real_s = Real_t>
using SphData = CylData<Real_t, Real_s>;

/// @brief Data for conical surfaces
/// @tparam Real_t Storage type
/// @tparam Real_s Interface type
template <typename Real_t, typename Real_s = Real_t>
struct ConeData {
  Real_t radius{0}; ///< Cone radus at Z = 0. Stored negative if flipped.
  Real_t slope{0};  ///< Cone slope  --> for cyl extension this would be 0

  ConeData() = default;
  ConeData(Real_s rad, Real_s slope, bool flip = false) : radius(flip ? -rad : rad), slope(slope) {}
  Real_s Radius() const { return std::abs(Real_s(radius)); }
  Real_s Slope() const { return slope; }
  bool IsFlipped() const { return radius < 0; }
};

} // namespace vgbrep

#endif