#pragma once

#include <unordered_map>
#include <vector>
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/robin_hood.h" // for fast hash map
#include <type_traits>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// A class representing a sparse voxel structure
// via a hash map; A voxel can hold an abitrary number of properties of type P
// If we have scalar properties.. we will save one indirection and attach the property
// directly to the map
template <typename P, bool ScalarProperties = false>
class FlatVoxelHashMap {
public:
  FlatVoxelHashMap(Vector3D<float> const &lower, Vector3D<float> const &Length, int Nx, int Ny, int Nz)
      : fNx{Nx}, fNy{Ny}, fNz{Nz}
  {
    fLowerX    = lower.x();
    fLowerY    = lower.y();
    fLowerZ    = lower.z();
    fDeltaX    = Length.x() / Nx;
    fDeltaY    = Length.y() / Ny;
    fDeltaZ    = Length.z() / Nz;
    fInvDeltaX = 1.f / fDeltaX;
    fInvDeltaY = 1.f / fDeltaY;
    fInvDeltaZ = 1.f / fDeltaZ;
  }

  // main method to add something; must be called in sorted key sequence
  // to achieve contiguous property storage per key
  void addProperty(Vector3D<float> const &point, P const &property)
  {
    const auto key = getVoxelKey(point);
    addPropertyForKey(key, property);
  }

  // length of diagonal of voxels
  float getVoxelDiagonal() const { return std::sqrt(fDeltaX * fDeltaX + fDeltaY * fDeltaY + fDeltaZ * fDeltaZ); }

  float getVoxelVolume() const { return fDeltaX * fDeltaY * fDeltaZ; }

  void addPropertyForKey(long key, P const &property)
  {
    auto iter = fVoxelMap.find(key);
    if (key == fLastKey) {
      // we append the property to the property vector
      // if a new property and increment the length count
      if (iter == fVoxelMap.end()) {
        std::cerr << "THERE SHOULD BE AN ENTRY ALREADY\n";
      }
      append<P>(iter, property);
    } else {
      // a new insertion; there should be no previous
      // entry at this key otherwise the property storage might
      // be corrupted
      if (iter != fVoxelMap.end()) {
        std::cerr << "FATAL ERROR IN INSERTION\n";
      }
      // inserting new property
      assign<P>(key, property);
      fLastKey = key;
    }
  }

  // a special method allowing to mark regions from a sequence of bounding boxes
  // TODO: somewhere else might be a better place for this
  void markFromBoundingBoxes(LogicalVolume const &lvol)
  {
    // IDEA:
    // a) iterate over the inside of the boxes in strides of
    //    out voxel grid and add the key and the property to some tmp storage

    // b) sort the keys so that we can insert them in order into the voxel container

    // get the BOUNDING BOXES
    std::vector<std::pair<long, P>> keyToProp;

    int nDaughters{0};
    auto abboxes = ABBoxManager::Instance().GetABBoxes(&lvol, nDaughters);
    for (int d = 0; d < nDaughters; ++d) {
      const auto &lower = abboxes[2 * d];
      const auto &upper = abboxes[2 * d + 1];

      int kxlow{-1}, kylow{-1}, kzlow{-1};
      int kxup{-1}, kyup{-1}, kzup{-1};
      getVoxelCoordinates(lower.x(), lower.y(), lower.z(), kxlow, kylow, kzlow);
      getVoxelCoordinates(upper.x(), upper.y(), upper.z(), kxup, kyup, kzup);
      // correct for out if bounds effects
      kxlow = std::max(kxlow, 0);
      kylow = std::max(kylow, 0);
      kzlow = std::max(kzlow, 0);
      // correct for out if bounds effects
      kxup = std::min(kxup, fNx - 1);
      kyup = std::min(kyup, fNy - 1);
      kzup = std::min(kzup, fNz - 1);

      // iterate over the box from lower to upper and fill keys
      for (int kx = kxlow; kx <= kxup; ++kx) {
        for (int ky = kylow; ky <= kyup; ++ky) {
          for (int kz = kzlow; kz <= kzup; ++kz) {
            keyToProp.push_back(std::pair<long, P>(getKeyFromCells(kx, ky, kz), d));
          }
        }
      }
    }

    // sort the container according to key
    std::sort(keyToProp.begin(), keyToProp.end(),
              [](std::pair<long, P> const &a, std::pair<long, P> const &b) { return a.first < b.first; });

    // iterate over final container and insert properties into the hash map
    for (auto &e : keyToProp) {
      addPropertyForKey(e.first, e.second);
    }
  }

  // main method to query something
  bool isOccupied(Vector3D<float> const &point) const { return fVoxelMap.find(getVoxelKey(point)) != fVoxelMap.end(); }

  bool isOccupied(long key) const { return fVoxelMap.find(key) != fVoxelMap.end(); }

  // get all voxel keys for a complete container of points
  // TODO: this can be nicely vectorized
  std::vector<long> getKeys(SOA3D<float> const &points)
  {
    std::vector<long> keys;
    for (size_t i = 0; i < points.size(); ++i) {
      keys.push_back(getKey(points.x(i), points.y(i), points.z(i)));
    }
    return keys;
  }

  // get all voxel keys for a complete container of points
  // TODO: this can be nicely vectorized
  template <typename T>
  void getKeys(SOA3D<T> const &points, std::vector<long> &keys)
  {
    for (size_t i = 0; i < points.size(); ++i) {
      keys.push_back(getKey(points.x(i), points.y(i), points.z(i)));
    }
  }

  // calculates the integer voxel coordinates
  void getVoxelCoordinates(float x, float y, float z, int &kx, int &ky, int &kz) const
  {
    kx = (int)((x - fLowerX) * fInvDeltaX);
    ky = (int)((y - fLowerY) * fInvDeltaY);
    kz = (int)((z - fLowerZ) * fInvDeltaZ);
  }

  long getKeyFromCells(int kx, int ky, int kz) const { return kx + fNx * (ky + kz * fNy); }

  long getKey(float x, float y, float z) const
  {
    const auto kx = (int)((x - fLowerX) * fInvDeltaX);
    const auto ky = (int)((y - fLowerY) * fInvDeltaY);
    const auto kz = (int)((z - fLowerZ) * fInvDeltaZ);
    // if (kx < 0 || kx >= fNx) {
    //  std::cerr << "key problem in x " << x << " (lowerX ) " << fLowerX << " key " << kx << "\n";
    //}
    // if (ky < 0 || ky >= fNy) {
    //  std::cerr << "key problem in y " << y << " (lowerY ) " << fLowerY << " key " << ky << "\n";
    //}
    // if (kz < 0 || kz >= fNz) {
    //  std::cerr << "key problem in z " << y << " (lowerZ ) " << fLowerZ << " key " << kz << "\n";
    //}
    return getKeyFromCells(kx, ky, kz);
  }

  long getVoxelKey(Vector3D<float> const &point) const { return getKey(point.x(), point.y(), point.z()); }

  const P *getProperties(Vector3D<float> const &point, int &length) const
  {
    return getPropertiesGivenKey(getVoxelKey(point), length);
  }

  // returns the pointer to the first property and how many properties
  // there are following
  // TODO: Instead of a pointer we should return an iterable
  // std::span as soon as this is available
  const P *getPropertiesGivenKey(long key, int &length) const { return getPropertiesImpl<P>(key, length); }

  Vector3D<int> keyToCell(long key) const
  {
    const auto kz = key / (fNx * fNy);
    const auto ky = (key - (kz * fNx * fNy)) / fNx;
    const auto kx = key - (kz * fNx * fNy) - ky * fNx;
    return Vector3D<int>(kx, ky, kz);
  }

  // returns **mid** point of voxel in cartesian coordinates
  Vector3D<float> keyToPos(long key) const
  {
    const auto kz = key / (fNx * fNy);
    const auto ky = (key - (kz * fNx * fNy)) / fNx;
    const auto kx = key - (kz * fNx * fNy) - ky * fNx;
    return Vector3D<float>(fLowerX + kx * fDeltaX * 1.5, fLowerY + ky * fDeltaY * 1.5, fLowerZ + kz * fDeltaZ * 1.5);
  }

  // return the lower and upper coordinates describing the extend of a voxel of some key
  void Extent(long key, Vector3D<float> &lower, Vector3D<float> &upper) const
  {
    const auto kz = key / (fNx * fNy);
    const auto ky = (key - (kz * fNx * fNy)) / fNx;
    const auto kx = key - (kz * fNx * fNy) - ky * fNx;
    lower         = Vector3D<float>(fLowerX + kx * fDeltaX, fLowerY + ky * fDeltaY, fLowerZ + kz * fDeltaZ);
    upper         = lower + Vector3D<float>(fDeltaX, fDeltaY, fDeltaZ);
  }

  // mainly for debugging
  void print() const
  {
    int count{0};
    int pcount{0};
    for (auto &k : fVoxelMap) {
      auto key = k.first;
      int number{0};
      auto props = getPropertiesGivenKey(key, number);
      pcount += number;
      count++;
      std::cerr << " voxel at key " << key << " : " << keyToCell(key) << " pos " << keyToPos(key) << " filled with "
                << number << " properties \n";
      std::cerr << "{ ";
      for (int i = 0; i < number; ++i) {
        std::cerr << props[i] << " , ";
      }
      std::cerr << " }\n";
    }
    std::cerr << "NUM VOXELS OCCUPIED " << count << " SUM PROPERTIES " << pcount << "\n";
  }

  /// clear the container
  void clear()
  {
    fVoxelMap.clear();
    fProperties.clear();
  }

  /// get number of filled voxels
  size_t size() const { return fVoxelMap.size(); }

private:
  // the dimensions
  int fNx = 0;
  int fNy = 0;
  int fNz = 0;

  // offset in 3D space
  float fLowerX    = 0.;
  float fLowerY    = 0.;
  float fLowerZ    = 0.;
  float fDeltaX    = 1.;
  float fDeltaY    = 1.;
  float fDeltaZ    = 1.;
  float fInvDeltaX = 1.;
  float fInvDeltaY = 1.;
  float fInvDeltaZ = 1.;

  long fLastKey = -1;

  // we map a long key to the start index of the property map together
  // with the number of properties for this voxel
  //  std::unordered_map<long, std::pair<int, int>> fVoxelMap;

  // this function gets evaluated only if ScalarProperties = true
  template <typename T>
  void assign(long key, typename std::enable_if<ScalarProperties, T>::type prop)
  {
    fVoxelMap[key] = prop;
  }

  // this function gets called/compiled only if ScalarProperties = false
  template <typename T>
  void assign(long key, typename std::enable_if<!ScalarProperties, T>::type prop)
  {
    fVoxelMap[key] = std::pair<int, int>(fProperties.size(), 1);
    fProperties.push_back(prop);
  }

  // this function gets evaluated only if ScalarProperties = true
  template <typename T, typename Iter>
  void append(Iter iter, typename std::enable_if<ScalarProperties, T>::type prop)
  {
    // nothing to be done - simply not possible
    // TODO: this has to throw an exception or assert
    std::cerr << "INVALID APPEND TO SCALAR PORPERTIES\n";
  }

  // this function gets called/compiled only if ScalarProperties = false
  template <typename T, typename Iter>
  void append(Iter iter, typename std::enable_if<!ScalarProperties, T>::type prop)
  {
    fProperties.push_back(prop);
    iter->second.second++; // we increment the number of properties for this voxel
  }

  // implementation for ScalarProperties = false
  template <typename T>
  T const *getPropertiesImpl(long key, int &length,
                             typename std::enable_if<!ScalarProperties, T>::type * = nullptr) const
  {
    length    = 0;
    auto iter = fVoxelMap.find(key);
    if (iter != fVoxelMap.end()) {
      length = iter->second.second;
      return &fProperties[iter->second.first];
    }
    return nullptr;
  }

  // implementation for ScalarProperties = true
  template <typename T>
  T const *getPropertiesImpl(long key, int &length,
                             typename std::enable_if<ScalarProperties, T>::type * = nullptr) const
  {
    length    = 0;
    auto iter = fVoxelMap.find(key);
    if (iter != fVoxelMap.end()) {
      length = 1;
      return &(iter->second);
    }
    return nullptr;
  }

  using MapValueType = typename std::conditional<ScalarProperties, P, std::pair<int, int>>::type;
  robin_hood::unordered_map<long, MapValueType> fVoxelMap;

  std::vector<P> fProperties;
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
