/// \file BVHsurfCreator.h
/// Creator of BVHsurf for a volume

#ifndef VECGEOM_SURFACES_BVHCREATOR_H_
#define VECGEOM_SURFACES_BVHCREATOR_H_

#include <fstream>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/bvh/BVHsurf.h>

namespace vgbrep {
namespace bvh {

enum class ConstructionAlgorithm : unsigned int {
  SplitLongestAxis         = 0,
  LargestDistanceAlongAxis = 1,
  SurfaceAreaHeuristic     = 2,
};

namespace {
using namespace vecgeom;
template <typename Real_b>
int *splitAlongLongestAxis(const AABBsurf<Real_b> *primitiveBoxes, int *begin, int *end,
                           const AABBsurf<Real_b> &currentBVHNode)
{
  const Vector3D<Precision> basis[] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  auto closestAxis                  = [](Vector3D<Precision> v) {
    v = v.Abs();
    return v[0] > v[2] ? (v[0] > v[1] ? 0 : 1) : (v[1] > v[2] ? 1 : 2);
  };
  Vector3D<Precision> p = currentBVHNode.Center();
  Vector3D<Precision> v = basis[closestAxis(currentBVHNode.Size())];

  return std::partition(begin, end, [&](size_t i) {
    Vector3D<Precision> center = static_cast<Vector3D<Precision>>(primitiveBoxes[i].Center());
    return Vector3D<Precision>::Dot(center - p, v) < 0.0;
  });
}

template <typename Real_b>
int *largestDistanceAlongAxis(const AABBsurf<Real_b> *primitiveBoxes, int *begin, int *end,
                              const AABBsurf<Real_b> & /*currentBVHNode*/)
{
  // Compute maximum extension of lower-left front corners along all axes
  float extension[3][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
  for (int axis = 0; axis <= 2; ++axis) {
    auto minMaxIt = std::minmax_element(
        begin, end, [=](size_t a, size_t b) { return primitiveBoxes[a].Min()[axis] < primitiveBoxes[b].Min()[axis]; });
    extension[axis][0] = primitiveBoxes[*minMaxIt.first].Min()[axis];
    extension[axis][1] = primitiveBoxes[*minMaxIt.second].Min()[axis];
  }

  const int splitAxis = std::distance(extension, std::max_element(extension, extension + 3, [](float a[], float b[]) {
                                        return a[1] - a[0] < b[1] - b[0];
                                      }));
  const float middlePoint = (extension[splitAxis][1] + extension[splitAxis][0]) / 2.f;
  return std::partition(begin, end, [=](size_t i) { return primitiveBoxes[i].Min()[splitAxis] < middlePoint; });
}

/**
 * In order to achieve stable splitting for the BVH, we cannot only sort by one axis. There needs
 * to be a strict order, so elements are not silently considered equal by the STL algorithms.
 * @param left,right Compute `left < right`.
 * @param sortAxis Principal axis to sort by.
 */
template <typename T>
bool less3D(const T &left, const T &right, const int sortAxis)
{
  return left[sortAxis] < right[sortAxis] ||
         (left[sortAxis] == right[sortAxis] && (left[(sortAxis + 1) % 3] < right[(sortAxis + 1) % 3] ||
                                                (left[(sortAxis + 1) % 3] == right[(sortAxis + 1) % 3] &&
                                                 left[(sortAxis + 2) % 3] < right[(sortAxis + 2) % 3])));
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
template <typename Real_b>
std::vector<std::pair<double, double>> sweepSurfaceArea(const AABBsurf<Real_b> *primitiveBoxes, int const *begin,
                                                        int const *end)
{
  if (begin >= end) return {};

  std::vector<std::pair<double, double>> areas(std::distance(begin, end), {0., 0.});

  AABBsurf<Real_b> box{primitiveBoxes[*begin]};
  for (auto it = begin + 1; it < end; ++it) {
    areas[it - begin].first = box.SurfaceArea();
    box                     = AABBsurf<Real_b>::Union(box, primitiveBoxes[*it]);
  }

  AABBsurf<Real_b> box2{primitiveBoxes[*(end - 1)]};
  for (auto it = end - 1; it >= begin; --it) {
    box2                     = AABBsurf<Real_b>::Union(box2, primitiveBoxes[*(it)]);
    areas[it - begin].second = box2.SurfaceArea();
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
template <typename Real_b>
int *surfaceAreaHeuristic(const AABBsurf<Real_b> *primitiveBoxes, int *begin, int *end,
                          const AABBsurf<Real_b> & /*currentBVHNode*/)
{
  int bestSplitAxis          = -1;
  double bestTraversalMetric = std::distance(begin, end);
  int bestSplitObject        = -1;
  const auto nObj            = std::distance(begin, end);

  int currentSortAxis = 0;
  auto sorter         = [primitiveBoxes, &currentSortAxis](int a, int b) {
    const auto centroidA   = primitiveBoxes[a].Center();
    const auto centroidB   = primitiveBoxes[b].Center();
    constexpr double shift = 0.01;
    return less3D(centroidA + shift * (centroidA - primitiveBoxes[a].Min()),
                          centroidB + shift * (centroidB - primitiveBoxes[b].Min()), currentSortAxis);
  };

  for (int axis = 0; axis <= 2; ++axis) {
    // Sort centroids along axis
    currentSortAxis = axis;
    std::sort(begin, end, sorter);

    // Sweep axis looking for best split
    const std::vector<std::pair<double, double>> surfaceSweep = sweepSurfaceArea(primitiveBoxes, begin, end);
    const auto totSurfArea                                    = surfaceSweep.front().second;

    for (int *splitObject = begin; splitObject < end; ++splitObject) {
      const auto left  = surfaceSweep[splitObject - begin].first / NonZero(totSurfArea);
      const auto right = surfaceSweep[splitObject - begin].second / NonZero(totSurfArea);
      assert(left <= 1. && right <= 1.);

      // Original heuristic
      const auto splitMetric = left * std::distance(begin, splitObject) + right * std::distance(splitObject, end) +
                               0.1 * abs(nObj / 2 - std::distance(begin, splitObject) / nObj); // Prefer balanced splits

      if (splitMetric < bestTraversalMetric) {
        bestTraversalMetric = splitMetric;
        bestSplitAxis       = axis;
        bestSplitObject     = *splitObject;
      }
    }
  }

  if (bestSplitAxis == -1) return end;

  currentSortAxis = bestSplitAxis;
  auto result = std::partition(begin, end, [sorter, bestSplitObject](size_t i) { return sorter(i, bestSplitObject); });

  return result;
}

/**
 * Array of splitting functions that can be used to construct the BVH tree.
 * @see ConstructionAlgorithm
 */
template <typename Real_b>
int *(*splittingFunction[])(const AABBsurf<Real_b> * /*primitveAABBs*/, int * /*firstPrimitive*/,
                            int * /*lastPrimitive*/, const AABBsurf<Real_b> & /*currentBVHNode*/) = {
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

template <typename Real_b>
void ComputeNodes(unsigned int id, int *first, int *last, unsigned int nodes, int *aPrimId, int *aNChild, int *aOffset,
                  AABBsurf<Real_b> *aNodes, AABBsurf<Real_b> *aAABBs, ConstructionAlgorithm constructionAlgorithm)
{
  if (id >= nodes) return;

  aNChild[id] = std::distance(first, last);
  aOffset[id] = std::distance(aPrimId, first);

  // Node without children. Stop recursing here.
  if (first == last) return;

  aNodes[id] = aAABBs[*first];
  for (auto it = std::next(first); it != last; ++it)
    aNodes[id] = AABBsurf<Real_b>::Union(aNodes[id], aAABBs[*it]);

  // Only one child. No need to continue
  if (std::next(first) == last) return;

  const auto algo = static_cast<unsigned int>(constructionAlgorithm);
  // assert(algo < sizeof(splittingFunction<Real_b>));

  int *pivot = splittingFunction<Real_b>[algo](aAABBs, first, last, aNodes[id]);
  assert(first <= pivot && pivot <= last);

  ComputeNodes(2 * id + 1, first, pivot, nodes, aPrimId, aNChild, aOffset, aNodes, aAABBs, constructionAlgorithm);
  ComputeNodes(2 * id + 2, pivot, last, nodes, aPrimId, aNChild, aOffset, aNodes, aAABBs, constructionAlgorithm);
}

template <typename Real_t>
static void InitBVH(int ivol, BVHsurf<typename vgbrep::SurfData<Real_t>::Real_b> &bvh,
                    vgbrep::CPUsurfData<Precision> const &cpudata, int depth = 0)
{
  using Real_b = typename vgbrep::SurfData<Real_t>::Real_b;
  uint aRootId = ivol;
  int n;
  /* ptr is a pointer to ndaughters times (min, max) corner vectors of each AABB */
  auto ptr = ABBoxManager<Real_b>::Instance().GetSurfaceABBoxes(ivol, n, cpudata);

  if (n <= 0) throw std::logic_error("Cannot construct BVH for volume with no surfaces!");

  auto aRootNChild = n;

  auto aAABBs = new AABBsurf<Real_b>[n];
  for (auto i = 0; i < n; ++i)
    aAABBs[i] = AABBsurf(ptr[2 * i], ptr[2 * i + 1]);

  /* Initialize map of primitive ids (i.e. child volume ids) as {0, 1, 2, ...}. */
  auto aPrimId = new int[n];
  std::iota(aPrimId, aPrimId + n, 0);

  /*
   * If depth = 0, choose depth dynamically based on the number of child volumes, up to the fixed
   * maximum depth. We use n/2 here because that creates a tree with roughly one node for every two
   * volumes, or roughly at most 4 children per leaf node. For example, for 1000 volumes, the
   * default depth would be log2(500) = 8.96 -> 8, with 2^8 - 1 = 511 nodes, and 256 leaf nodes.
   */
  int aDepth = std::min(depth ? depth : std::max(0, (int)std::log2(n / 2)), BVHsurf<Real_b>::BVH_MAX_DEPTH);

  unsigned int nodes = (2 << aDepth) - 1;

  auto aNChild = new int[nodes];
  auto aOffset = new int[nodes];
  auto aNodes  = new AABBsurf<Real_b>[nodes];
  std::fill(aNChild, aNChild + nodes, 0);
  std::fill(aOffset, aOffset + nodes, -1);

  /* Recursively initialize BVH nodes starting at the root node */
  ComputeNodes(0, aPrimId, aPrimId + n, nodes, aPrimId, aNChild, aOffset, aNodes, aAABBs,
               ConstructionAlgorithm::SurfaceAreaHeuristic);

  /* Mark internal nodes with a negative number of children to simplify traversal */
  for (unsigned int id = 0; id < nodes / 2; ++id)
    if (aNChild[id] > 8 && (aNChild[id] == aNChild[2 * id + 1] + aNChild[2 * id + 2])) aNChild[id] = -1;

  /* Fill the BVH */
  bvh.Set(aRootId, aRootNChild, aDepth, aPrimId, aAABBs, aOffset, aNChild, aNodes);
}

// Dump the BVH in binary format
template <typename Real_b>
static void DumpBVH(const BVHsurf<Real_b> &bvh, const char *filename)
{
  auto fillAABBbuffer = [](const AABBsurf<Real_b> *aabbs, int n, double *buffer) {
    for (auto i = 0; i < n; ++i) {
      buffer[6 * i]     = aabbs[i].fMin[0];
      buffer[6 * i + 1] = aabbs[i].fMin[1];
      buffer[6 * i + 2] = aabbs[i].fMin[2];
      buffer[6 * i + 3] = aabbs[i].fMax[0];
      buffer[6 * i + 4] = aabbs[i].fMax[1];
      buffer[6 * i + 5] = aabbs[i].fMax[2];
    }
  };

  std::ofstream stm(filename, std::ios::binary);
  if (!stm.is_open()) {
    std::cout << "bad file " << filename << "\n";
    return;
  }
  auto aRootId = bvh.GetRootId();
  stm.write(reinterpret_cast<const char *>(&aRootId), sizeof(aRootId));
  auto aRootNChild = bvh.GetRootNChild();
  stm.write(reinterpret_cast<const char *>(&aRootNChild), sizeof(aRootNChild));
  auto aDepth = bvh.GetDepth();
  stm.write(reinterpret_cast<const char *>(&aDepth), sizeof(aDepth));
  unsigned int nodes = (2 << aDepth) - 1;
  auto aPrimId       = bvh.GetPrimId();
  stm.write(reinterpret_cast<const char *>(aPrimId), aRootNChild * sizeof(int));
  auto aOffset = bvh.GetOffset();
  stm.write(reinterpret_cast<const char *>(aOffset), nodes * sizeof(int));
  auto aNChild = bvh.GetNChild();
  stm.write(reinterpret_cast<const char *>(aNChild), nodes * sizeof(int));
  double *buffer = new double[6 * (nodes + aRootNChild)];
  fillAABBbuffer(bvh.GetNodes(), nodes, buffer);
  stm.write(reinterpret_cast<const char *>(buffer), 6 * nodes * sizeof(double));
  fillAABBbuffer(bvh.GetAABBs(), aRootNChild, buffer + 6 * nodes);
  stm.write(reinterpret_cast<const char *>(buffer + 6 * nodes), 6 * aRootNChild * sizeof(double));
  stm.close();
  delete[] buffer;
}

} // namespace bvh
} // namespace vgbrep

#endif
