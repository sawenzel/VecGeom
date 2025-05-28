#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/AOS3D.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/PolyhedronStruct.h"
#include "VecGeom/volumes/PolyconeStruct.h"
#include <iostream>

using namespace vecgeom;

template <typename T, template <typename> class ContainerType>
void SizeTest()
{
  ContainerType<T> container(0);
  VECGEOM_ASSERT(container.size() == 0);
  VECGEOM_ASSERT(container.capacity() == 0);
  container.reserve(8);
  VECGEOM_ASSERT(container.size() == 0);
  VECGEOM_ASSERT(container.capacity() == 8);
  container.resize(6);
  VECGEOM_ASSERT(container.size() == 6);
  VECGEOM_ASSERT(container.capacity() == 8);
  container.set(3, Vector3D<T>(1, 2, 3));
  container.set(2, 3, 2, 1);
  VECGEOM_ASSERT(container[3] == Vector3D<T>(1, 2, 3));
  VECGEOM_ASSERT(container[2] == Vector3D<T>(3, 2, 1));
  VECGEOM_ASSERT(container.z(3) == 3);
  VECGEOM_ASSERT(container.x(2) == 3);
}

void VectorTest()
{
  auto equal_vect = [](Vector3D<double> const &v1, Vector3D<double> const &v2) {
    return (v1.x() == v2.x() && v1.y() == v2.y() && v1.z() == v2.z());
  };
  Vector<Vector3D<double>> container(0); // {}
  VECGEOM_ASSERT(container.size() == 0);
  VECGEOM_ASSERT(container.capacity() == 0);
  container.reserve(8); // { {}, {}, {}, {}, {}, {}, {}, {} }
  VECGEOM_ASSERT(container.size() == 0);
  VECGEOM_ASSERT(container.capacity() == 8);
  container.resize(2, {}); // { {0}, {0}, {}, {}, {}, {}, {}, {} }
  VECGEOM_ASSERT(container.size() == 2);
  VECGEOM_ASSERT(container.capacity() == 8);
  container.push_back({1., 2., 3.}); // { {0}, {0}, {1,2,3}, {}, {}, {}, {}, {} }
  VECGEOM_ASSERT(equal_vect(container[0], {}));
  VECGEOM_ASSERT(equal_vect(container[2], {1., 2., 3.}));

  // Assignment operator
  Vector<Vector3D<double>> container1;
  container1 = container; // { {0}, {0}, {1,2,3} }
  VECGEOM_ASSERT(container1.size() == 3);
  VECGEOM_ASSERT(container1.capacity() == 3);
  VECGEOM_ASSERT(equal_vect(container1[2], {1., 2., 3.}));
  container1.reserve(4); // { {0}, {0}, {1,2,3}, {} }
  VECGEOM_ASSERT(container1.size() == 3);
  VECGEOM_ASSERT(container1.capacity() == 4);
  VECGEOM_ASSERT(equal_vect(container1[2], {1., 2., 3.}));
  container1.push_back({1., 2., 3.}); // { {0}, {0}, {1,2,3}, {1,2,3} }
  VECGEOM_ASSERT(container1.size() == 4);
  container1.push_back({3., 2., 1.}); // { {0}, {0}, {1,2,3}, {1,2,3}, {3,2,1}, {}, {}, {} }
  VECGEOM_ASSERT(container1.size() == 5);
  VECGEOM_ASSERT(container1.capacity() == 8);

  // initializer list, assignment
  Vector<Vector3D<double>> container2 = {{}, {}, {1., 2., 3.}, {3., 2., 1.}};
  VECGEOM_ASSERT(container2.size() == 4);
  VECGEOM_ASSERT(container2.capacity() == 4);
  VECGEOM_ASSERT(equal_vect(container2[3], {3., 2., 1.}));
  container1 = container2;
  VECGEOM_ASSERT(container1.size() == 4);
  VECGEOM_ASSERT(container1.capacity() == 8);
  VECGEOM_ASSERT(equal_vect(container1[3], {3., 2., 1.}));

  // Fixed size
  Vector3D<double> arr[3] = {{}, {1., 2., 3.}, {3., 2., 1.}};
  Vector<Vector3D<double>> container3(arr, 3);
  VECGEOM_ASSERT(container3.size() == 3);
  VECGEOM_ASSERT(container3.capacity() == 3);
  VECGEOM_ASSERT(!container3.is_allocated());
  VECGEOM_ASSERT(equal_vect(container3[2], {3., 2., 1.}));
  container1.erase(container1.end() - 1);
  container3 = container1; //-> this will fire an assert without the erase before
  VECGEOM_ASSERT(equal_vect(container3[2], {1., 2., 3.}));

  // Copy constructor
  Vector<Vector3D<double>> container4(container3);
  VECGEOM_ASSERT(container4.size() == 3);
  VECGEOM_ASSERT(container4.capacity() == 3);
  VECGEOM_ASSERT(equal_vect(container4[2], {1., 2., 3.}));
}

template <typename T, template <typename> class ContainerType>
void AllocationTest()
{
  constexpr size_t nz     = 4;
  constexpr size_t nside  = 4;
  Precision RMINVec3[nz]  = {1., 0.5, 0., 0.};
  Precision RMAXVec3[nz]  = {2., 1., 2., 2.};
  Precision Z_Values3[nz] = {-1., 0., 0., 1.};

  Precision sphi3 = -kPi / 4;
  Precision dphi3 = 2 * kPi;

  PolyhedronStruct<Precision> ph1(sphi3, dphi3, nside, nz, Z_Values3, RMINVec3, RMAXVec3);
  auto buffer_size = AlignedAllocator::aligned_sizeof<PolyhedronStruct<Precision>>(1, kAlignmentBoundary, sphi3, dphi3,
                                                                                   4, 4, Z_Values3, RMINVec3, RMAXVec3);
  printf("PolyhedronStruct size: %lu bytes\n", buffer_size);
  ph1.Dump();
  char *buffer = new char[buffer_size];
  AlignedAllocator a(buffer, buffer_size);
  printf("Created allocator in buffer from %p to %p, size %lu bytes\n", a.address, (void *)((size_t)a.address + a.sz),
         a.sz);
  auto ph2 = a.aligned_alloc<PolyhedronStruct<Precision>>(1, kAlignmentBoundary, sphi3, dphi3, 4, 4, Z_Values3,
                                                          RMINVec3, RMAXVec3, a);
  ph2->Dump();
  delete[] buffer;

  PolyconeStruct<Precision> *pc1 = new PolyconeStruct<Precision>();
  pc1->Init(sphi3, dphi3, 4, Z_Values3, RMINVec3, RMAXVec3);
  pc1->Dump();

  auto num_sections = pc1->fSections.size();
  buffer_size       = AlignedAllocator::aligned_sizeof<PolyconeStruct<Precision>>(1, 0, num_sections);
  buffer            = new char[buffer_size];
  AlignedAllocator a1(buffer, buffer_size);
  printf("Created allocator in buffer from %p to %p, size %lu bytes\n", a1.address,
         (void *)((size_t)a1.address + a1.sz), a1.sz);
  std::vector<double> z_values, rmin_values, rmax_values;
  pc1->ReconstructSectionArrays(z_values, rmin_values, rmax_values);
  VECGEOM_ASSERT(z_values.size() == nz);

  auto pc2 = a1.aligned_alloc<PolyconeStruct<Precision>>(
      1, 0, pc1->fEqualRmax, pc1->fContinuityOverAll, pc1->fConvexityPossible, pc1->fStartPhi, pc1->fDeltaPhi,
      num_sections, pc1->fNz, z_values.data(), rmin_values.data(), rmax_values.data(), a1);
  pc2->Dump();

  delete pc1;
  delete[] buffer;
}

int main()
{
  SizeTest<Precision, AOS3D>();
  SizeTest<Precision, SOA3D>();
  VectorTest();
  // AllocationTest<Precision, SOA3D>();
  return 0;
}
