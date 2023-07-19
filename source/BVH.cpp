/// \file BVH.cpp
/// \author Guilherme Amadio

#include "VecGeom/base/BVH.h"

#include "VecGeom/management/ABBoxManager.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

constexpr int BVH::BVH_MAX_DEPTH;

enum class BVH::ConstructionAlgorithm : unsigned int {
  SplitLongestAxis = 0,
  LargestDistanceAlongAxis = 1,
  SurfaceAreaHeuristic = 2,
};

/**
 * \class BVH
 * The BVH class is based on a complete binary tree stored contiguously in an array.
 *
 * For a given node id, 2*id + 1 gives the left child id, and 2*id + 2 gives the right child id.
 * For example, node 0 has its children at positions 1 and 2 in the vector, and for node 2, the
 * child nodes are at positions 5 and 6. The tree has 1 + 2 + 4 + ... + d nodes in total, where
 * d is the depth of the tree, or 1, 3, 7, ..., 2^d - 1 nodes in total. Visually, the ids for each
 * node look like shown below for a tree of depth 3:
 *
 *                                              0
 *                                             / \
 *                                            1   2
 *                                           / \ / \
 *                                          3  4 5  6
 *
 * with 2^3 - 1 = 7 nodes in total. For each node id, fNChild[id] gives the number of children of
 * the logical volume that belong to that node, and fOffset[id] gives the offset in the id map
 * fPrimId where the ids of the child volumes are stored, such that accessing fPrimId[fOffset[id]]
 * gives the first child id, then fPrimId[fOffset[id] + 1] gives the second child, up to fNChild[id]
 * children. The bounding boxes stored in fAABBs are in the original order, so they are accessed by
 * the original child number (i.e. the id stored in fPrimId, not by a node id of the tree itself).
 */

BVH::BVH(LogicalVolume const &volume, int depth) : fLV(volume)
{
  int n;

  /* ptr is a pointer to ndaughters times (min, max) corner vectors of each AABB */
  Vector3D<Precision> *ptr = ABBoxManager::Instance().GetABBoxes(&volume, n);

  if (n <= 0) throw std::logic_error("Cannot construct BVH for volume with no children!");

  fAABBs = new AABB[n];
  for (int i = 0; i < n; ++i)
    fAABBs[i] = AABB(ptr[2 * i], ptr[2 * i + 1]);

  /* Initialize map of primitive ids (i.e. child volume ids) as {0, 1, 2, ...}. */
  fPrimId = new int[n];
  std::iota(fPrimId, fPrimId + n, 0);

  /*
   * If depth = 0, choose depth dynamically based on the number of child volumes, up to the fixed
   * maximum depth. We use n/2 here because that creates a tree with roughly one node for every two
   * volumes, or roughly at most 4 children per leaf node. For example, for 1000 volumes, the
   * default depth would be log2(500) = 8.96 -> 8, with 2^8 - 1 = 511 nodes, and 256 leaf nodes.
   */
  fDepth = std::min(depth ? depth : std::max(0, (int)std::log2(n / 2)), BVH_MAX_DEPTH);

  unsigned int nodes = (2 << fDepth) - 1;

  fNChild = new int[nodes];
  fOffset = new int[nodes];
  fNodes  = new AABB[nodes];
  std::fill(fNChild, fNChild+nodes, 0);
  std::fill(fOffset, fOffset+nodes, -1);

  /* Recursively initialize BVH nodes starting at the root node */
  ComputeNodes(0, fPrimId, fPrimId + n, nodes, ConstructionAlgorithm::SurfaceAreaHeuristic);

  /* Mark internal nodes with a negative number of children to simplify traversal */
  for (unsigned int id = 0; id < nodes / 2; ++id)
    if (fNChild[id] > 8 && (fNChild[id] == fNChild[2 * id + 1] + fNChild[2 * id + 2])) fNChild[id] = -1;
}

#ifdef VECGEOM_ENABLE_CUDA
VECCORE_ATT_DEVICE
BVH::BVH(LogicalVolume const *volume, int depth, int *dPrimId, AABB *dAABBs, int *dOffset, int *dNChild, AABB *dNodes)
    : fLV(*volume), fPrimId(dPrimId), fOffset(dOffset), fNChild(dNChild), fNodes(dNodes), fAABBs(dAABBs), fDepth(depth)
{
}
#endif

VECCORE_ATT_HOST_DEVICE
void BVH::Print(bool verbose) const
{
  printf("\nBVH(%u): addr: %p, depth: %d, nodes: %d, children: %zu, name: %s\n",
         fLV.id(), this, fDepth, (2 << fDepth) - 1, fLV.GetDaughters().size(),
         fLV.GetName() );
  if (verbose) {
    constexpr auto width = 4;
    int nChildToPad = 1;
    for (int depth = fDepth; depth >= 0; --depth) {
      const auto begin = (1 << depth) - 1;
      const auto end   = (2 << depth) - 1;
      for (int node = begin; node < end; ++node) {
        if (nChildToPad > 1) printf("%*c", (nChildToPad-1)*width/2, ' ');
        printf("%3d ", fNChild[node]);
        if (nChildToPad > 1) printf("%*c", (nChildToPad-1)*width/2, ' ');
      }
      printf("\n");
      nChildToPad *= 2;
    }
  }
}

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::BVH> BVH::CopyToGpu(void *addr) const
{
  int *dPrimId;
  int *dOffset;
  int *dNChild;
  cuda::AABB *dAABBs;
  cuda::AABB *dNodes;

  if (!addr) throw std::logic_error("Cannot copy BVH into a null pointer!");

  int n = fLV.GetDaughters().size();

  CudaCheckError(CudaMalloc((void **)&dPrimId, n * sizeof(int)));
  CudaCheckError(CudaMalloc((void **)&dAABBs, n * sizeof(AABB)));

  CudaCheckError(CudaCopyToDevice((void *)dPrimId, (void *)fPrimId, n * sizeof(int)));
  CudaCheckError(CudaCopyToDevice((void *)dAABBs, (void *)fAABBs, n * sizeof(AABB)));

  int nodes = (2 << fDepth) - 1;

  CudaCheckError(CudaMalloc((void **)&dOffset, nodes * sizeof(int)));
  CudaCheckError(CudaMalloc((void **)&dNChild, nodes * sizeof(int)));
  CudaCheckError(CudaMalloc((void **)&dNodes, nodes * sizeof(AABB)));

  CudaCheckError(CudaCopyToDevice((void *)dOffset, (void *)fOffset, nodes * sizeof(int)));
  CudaCheckError(CudaCopyToDevice((void *)dNChild, (void *)fNChild, nodes * sizeof(int)));
  CudaCheckError(CudaCopyToDevice((void *)dNodes, (void *)fNodes, nodes * sizeof(AABB)));

  cuda::LogicalVolume const *dvolume = CudaManager::Instance().LookupLogical(&fLV).GetPtr();

  if (!dvolume) {
    std::cerr << "Failed for lv " << fLV.GetLabel() << " (id = " << fLV.id() << ")" << std::endl;
    throw std::logic_error("Cannot copy BVH because logical volume does not exist on the device.");
  }

  DevicePtr<cuda::BVH> dBVH(addr);

  dBVH.Construct(dvolume, fDepth, dPrimId, dAABBs, dOffset, dNChild, dNodes);

  return dBVH;
}
#endif

BVH::~BVH()
{
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
  if (fPrimId) delete[] fPrimId;
  if (fOffset) delete[] fOffset;
  if (fNChild) delete[] fNChild;
  if (fNodes) delete[] fNodes;
  if (fAABBs) delete[] fAABBs;
#endif
}

namespace {
int ClosestAxis(Vector3D<Precision> v)
{
  v = v.Abs();
  return v[0] > v[2] ? (v[0] > v[1] ? 0 : 1) : (v[1] > v[2] ? 1 : 2);
}

int * splitAlongLongestAxis(const AABB * primitiveBoxes,
                            int * begin, int * end,
                            const AABB & currentBVHNode) {
  const Vector3D<Precision> basis[] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  Vector3D<Precision> p = currentBVHNode.Center();
  Vector3D<Precision> v = basis[ClosestAxis(currentBVHNode.Size())];

  return std::partition(begin, end, [&](size_t i) { return Vector3D<Precision>::Dot(primitiveBoxes[i].Center() - p, v) < 0.0; });
}

int * largestDistanceAlongAxis(const AABB * primitiveBoxes,
                               int * begin, int * end,
                               const AABB & /*currentBVHNode*/) {
  // Compute maximum extension of lower-left front corners along all axes
  float extension[3][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
  for (int axis = 0; axis <= 2; ++axis) {
    auto minMaxIt = std::minmax_element(begin, end, [=](size_t a, size_t b){
      return primitiveBoxes[a].Min()[axis] < primitiveBoxes[b].Min()[axis];
    });
    extension[axis][0] = primitiveBoxes[*minMaxIt.first].Min()[axis];
    extension[axis][1] = primitiveBoxes[*minMaxIt.second].Min()[axis];
  }

  const int splitAxis = std::distance(extension, std::max_element(extension, extension+3, [](float a[], float b[]){
    return a[1]-a[0] < b[1]-b[0];
  }));
  const float middlePoint = (extension[splitAxis][1] + extension[splitAxis][0]) / 2.;
  return std::partition(begin, end, [=](size_t i) {
    return primitiveBoxes[i].Min()[splitAxis] < middlePoint;
  });
}

/**
 * In order to achieve stable splitting for the BVH, we cannot only sort by one axis. There needs
 * to be a strict order, so elements are not silently considered equal by the STL algorithms.
 * @param left,right Compute `left < right`.
 * @param sortAxis Principal axis to sort by.
 */
template<typename T>
bool less3D(const T & left, const T & right, const int sortAxis) {
  return left[sortAxis] < right[sortAxis] ||
    ( left[sortAxis] == right[sortAxis] &&
      ( left[(sortAxis+1)%3] < right[(sortAxis+1)%3] ||
        ( left[(sortAxis+1)%3] == right[(sortAxis+1)%3] && left[(sortAxis+2)%3] < right[(sortAxis+2)%3] )
      )
    );
}

/**
 * Compute the surface areas of bounding boxes that surround the given primitives,
 * sweeping from left to right and vice-versa.
 *
 * For three objects 0 1 2, the vector contains the following surface areas:
 * ( | 0+1+2)  (0 | 1+2)   (0+1 | 2)
 *
 * That is, if the object N is intended to be the pivot object, the surface area of
 * - everything left of N is `areas[N].first`
 * - everything right of N + N itself is `areas[N].second`
 *
 * @param primitiveBoxes Array of bounding boxes of primitives.
 * @param begin Index of first primitive to be considered.
 * @param end   Past-the-end index of primitives to be considered.
 */
std::vector<std::pair<double, double>> sweepSurfaceArea(const AABB * primitiveBoxes, int const * begin, int const * end) {
  if (begin >= end) return {};

  std::vector<std::pair<double, double>> areas(std::distance(begin, end), {0., 0.});

  AABB box{primitiveBoxes[*begin]};
  for (auto it = begin+1; it < end; ++it) {
    areas[it-begin].first = box.SurfaceArea();
    box = AABB::Union(box, primitiveBoxes[*it]);
  }

  AABB box2{primitiveBoxes[*(end-1)]};
  for (auto it = end - 1; it >= begin; --it) {
    box2 = AABB::Union(box2, primitiveBoxes[*(it)]);
    areas[it-begin].second = box2.SurfaceArea();
  }

  return areas;
}

/**
 * Use the surface area heuristic to construct a BVH tree.
 * This algorithm tries to split the primitives such that they form clusters that have a minimal surface
 * area, as this decreases the likelihood that a BVH node is intersected by a ray.
 * Contrary to what's used in standard graphics, the cost function has an additional term that prevents
 * very large clusters. For the conventional SAH, a long line of equally spaced primitives would does not
 * yield an obvious splitting point, as all splits lead to the same total surface area for both child nodes.
 * To prevent this, there is an extra term, which will encourage a 50:50 split.
 * @param primitveBoxes Array of bounding boxes of primitives.
 * @param begin Index of first primitive to be considered.
 * @param end   Past-the-end index of primitives to be considered.
 * @return Index of the first element of the second group. If this is `end`, no good split was found.
 */
int * surfaceAreaHeuristic(const AABB * primitiveBoxes,
                          int * begin, int * end,
                          const AABB & /*currentBVHNode*/) {
  int bestSplitAxis = -1;
  double bestTraversalMetric = std::distance(begin, end);
  int bestSplitObject = -1;
  const auto nObj = std::distance(begin, end);

  int currentSortAxis = 0;
  auto sorter = [primitiveBoxes, &currentSortAxis](int a, int b) {
    const auto centroidA = primitiveBoxes[a].Center();
    const auto centroidB = primitiveBoxes[b].Center();
    constexpr double shift = 0.01;
    return less3D(centroidA + shift*(centroidA - primitiveBoxes[a].Min()),
                  centroidB + shift*(centroidB - primitiveBoxes[b].Min()),
                  currentSortAxis);
  };

  for (int axis = 0; axis <= 2; ++axis) {
    // Sort centroids along axis
    currentSortAxis = axis;
    std::sort(begin, end, sorter);

    // Sweep axis looking for best split
    const std::vector<std::pair<double,double>> surfaceSweep = sweepSurfaceArea(primitiveBoxes, begin, end);
    const auto totSurfArea = surfaceSweep.front().second;

    for (int * splitObject = begin; splitObject < end; ++splitObject) {
      const auto left  = surfaceSweep[splitObject-begin].first/totSurfArea;
      const auto right = surfaceSweep[splitObject-begin].second/totSurfArea;
      assert(left <= 1. && right <= 1.);

      const auto splitMetric =
            left * std::distance(begin, splitObject)
          + right * std::distance(splitObject, end)
          + 0.1 * abs(nObj/2 - std::distance(begin, splitObject) / nObj); // Prefer balanced splits

      if (splitMetric < bestTraversalMetric) {
        bestTraversalMetric = splitMetric;
        bestSplitAxis = axis;
        bestSplitObject = *splitObject;
      }
    }
  }

  if (bestSplitAxis == -1)
    return end;

  currentSortAxis = bestSplitAxis;
  auto result = std::partition(begin, end, [sorter,bestSplitObject](size_t i) {
    return sorter(i, bestSplitObject);
  });

  return result;
}

/**
 * Array of splitting functions that can be used to construct the BVH tree.
 * @see BVH::ConstructionAlgorithm
 */
int * (*splittingFunction[])(const AABB * /*primitveAABBs*/,
                            int * /*firstPrimitive*/, int * /*lastPrimitive*/,
                            const AABB & /*currentBVHNode*/) = {
  &splitAlongLongestAxis,
  &largestDistanceAlongAxis,
  &surfaceAreaHeuristic,
};

} // anonymous namespace

/*
 * BVH::ComputeNodes() initializes nodes of the BVH. It first computes the number of children that
 * belong to the current node based on the iterator range that is passed as input, as well as the
 * offset where the children of this node start. Then, it computes the overall bounding box of the
 * current node as the union of all bounding boxes of its child volumes. Then, if recursion should
 * continue, a splitting plane is chosen based on the longest dimension of the bounding box for the
 * current node, and the children are sorted such that all children on each side of the splitting
 * plane are stored contiguously. Then the function is called recursively with the iterator
 * sub-ranges for volumes on each side of the splitting plane to construct its left and right child
 * nodes. Recursion stops if a child node is deeper than the maximum depth, if the iterator range
 * is empty (i.e. no volumes on this node, maybe because all child volumes' centroids are on the
 * same side of the splitting plane), or if the node contains only a single volume.
 */

void BVH::ComputeNodes(unsigned int id, int *first, int *last, unsigned int nodes,
                       BVH::ConstructionAlgorithm constructionAlgorithm)
{
  if (id >= nodes) return;

  fNChild[id] = std::distance(first, last);
  fOffset[id] = std::distance(fPrimId, first);

  // Node without children. Stop recursing here.
  if (first == last) return;

  fNodes[id] = fAABBs[*first];
  for (auto it = std::next(first); it != last; ++it)
    fNodes[id] = AABB::Union(fNodes[id], fAABBs[*it]);

  // Only one child. No need to continue
  if (std::next(first) == last) return;

  const auto algo = static_cast<unsigned int>(constructionAlgorithm);
  assert(algo < sizeof(splittingFunction));

  int * pivot = splittingFunction[algo](fAABBs, first, last, fNodes[id]);
  assert(first <= pivot && pivot <= last);

  ComputeNodes(2 * id + 1, first, pivot, nodes, constructionAlgorithm);
  ComputeNodes(2 * id + 2, pivot, last,  nodes, constructionAlgorithm);
}

/*
 * BVH::ApproachNextDaughter is very similar to CheckDaughterIntersections but computes the first
 * hit daughter bounding box instead of the next hit shape. This lighter computation is used to
 * first approach the next hit solid before computing the actual distance, in the attempt to
 * reduce the numerical rounding error due to propagation to boundary.
 */

VECCORE_ATT_HOST_DEVICE
void BVH::ApproachNextDaughter(Vector3D<Precision> point, Vector3D<Precision> dir, Precision &step,
                               VPlacedVolume const *last) const
{
  unsigned int stack[BVH_MAX_DEPTH] = {0}, *ptr = &stack[1];

  /* Calculate and reuse inverse direction to save on divisions */
  Vector3D<Precision> invdir(1.0 / NonZero(dir[0]), 1.0 / NonZero(dir[1]), 1.0 / NonZero(dir[2]));

  do {
    unsigned int id = *--ptr; /* pop next node id to be checked from the stack */

    if (fNChild[id] >= 0) {
      /* For leaf nodes, loop over children */
      for (int i = 0; i < fNChild[id]; ++i) {
        int prim = fPrimId[fOffset[id] + i];
        /* Check AABB first, then the volume itself if needed */
        if (fAABBs[prim].IntersectInvDir(point, invdir, step)) {
          auto vol  = fLV.GetDaughters()[prim];
          // Convert point/direction to daughter frame
          Transformation3D const *tr     = vol->GetTransformation();
          Vector3D<Precision> localpoint = tr->Transform(point);
          Vector3D<Precision> localdir   = tr->TransformDirection(dir);
          Vector3D<Precision> invlocaldir(1.0 / NonZero(localdir[0]), 1.0 / NonZero(localdir[1]), 1.0 / NonZero(localdir[2]));
          auto dist = vol->GetUnplacedVolume()->ApproachSolid(localpoint, invlocaldir);
          /* If distance to current child is smaller than current step, update step and hitcandidate */
          if (dist < step && !(dist <= 0.0 && vol == last)) step = dist;
        }
      }
    } else {
      unsigned int childL = 2 * id + 1;
      unsigned int childR = 2 * id + 2;

      /* For internal nodes, check AABBs to know if we need to traverse left and right children */
      Precision tminL = kInfLength, tmaxL = -kInfLength, tminR = kInfLength, tmaxR = -kInfLength;

      fNodes[childL].ComputeIntersectionInvDir(point, invdir, tminL, tmaxL);
      fNodes[childR].ComputeIntersectionInvDir(point, invdir, tminR, tmaxR);

      bool traverseL = tminL <= tmaxL && tmaxL >= 0.0 && tminL < step;
      bool traverseR = tminR <= tmaxR && tmaxR >= 0.0 && tminR < step;

      /*
       * If both left and right nodes need to be checked, check closest one first.
       * This ensures step gets short as fast as possible so we can skip more nodes without checking.
       */
      if (tminR < tminL) {
        if (traverseR) *ptr++ = childR;
        if (traverseL) *ptr++ = childL;
      } else {
        if (traverseL) *ptr++ = childL;
        if (traverseR) *ptr++ = childR;
      }
    }
  } while (ptr > stack);
}


} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA
namespace cxx {
template void DevicePtr<cuda::BVH>::Construct(cuda::LogicalVolume const *volume, int depth, int *dPrimId,
                                              cuda::AABB *dAABBs, int *dOffset, int *dNChild, cuda::AABB *dNodes) const;
} // namespace cxx
#endif

} // namespace vecgeom
