/*
 * EmbreeManager.cpp
 *
 *  Created on: May 18, 2018
 *      Author: swenzel
 */

#include "VecGeom/management/EmbreeManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/ABBoxManager.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Stopwatch.h"
#include <embree3/rtcore.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

double g_step;
LogicalVolume const *g_lvol;
VPlacedVolume const *g_pvol;
VPlacedVolume const *g_lastexited;
Vector3D<double> const *g_pos;
Vector3D<double> const *g_dir;
Vector3D<float> const *g_normals;
int g_count;
// bool* g_geomIDs;
std::vector<int> *g_geomIDs = new std::vector<int>;
EmbreeManager::BoxIdDistancePair_t *g_hitlist;

void EmbreeManager::InitStructure(LogicalVolume const *lvol)
{
  auto numregisteredlvols = GeoManager::Instance().GetRegisteredVolumesCount();
  if (fStructureHolder.size() != numregisteredlvols) {
    fStructureHolder.resize(numregisteredlvols, nullptr);
  }
  if (fStructureHolder[lvol->id()] != nullptr) {
    RemoveStructure(lvol);
  }
  BuildStructure(lvol);
}

void EmbreeManager::BuildStructure(LogicalVolume const *vol)
{
  // for a logical volume we are referring to the functions that builds everything giving just bounding
  // boxes
  int nDaughters{0};

  if (fBuildMode == EmbreeBuildMode::kAABBox) {
    // get the boxes (and number of boxes), must be called before the BuildStructure
    // function call since otherwise nDaughters is not guaranteed to be initialized
    auto boxes                  = ABBoxManager<Precision>::Instance().GetABBoxes(vol, nDaughters);
    auto structure              = BuildStructureFromBoundingBoxes(boxes, nDaughters);
    fStructureHolder[vol->id()] = structure;
    VECGEOM_ASSERT((int)vol->GetDaughters().size() == nDaughters);
  } else if (fBuildMode == EmbreeBuildMode::kBBox) {
    auto structure              = BuildStructureFromBoundingBoxes(vol);
    fStructureHolder[vol->id()] = structure;
  }
}

EmbreeManager::EmbreeAccelerationStructure *EmbreeManager::BuildStructureFromBoundingBoxes(
    LogicalVolume const *lvol) const
{
  Stopwatch timer;
  timer.Start();
  auto device = rtcNewDevice(""); // --> could be a global device??
  auto scene  = rtcNewScene(device);
  rtcSetSceneFlags(scene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
  rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

  const auto daughters         = lvol->GetDaughtersp();
  const auto numberofdaughters = daughters->size();

  auto structure            = new EmbreeManager::EmbreeAccelerationStructure();
  structure->fDevice        = device;
  structure->fScene         = scene;
  structure->fNormals       = new Vector3D<float>[numberofdaughters * 6];
  structure->fNumberObjects = numberofdaughters;

  for (size_t d = 0; d < numberofdaughters; ++d) {
    const auto pvol = daughters->operator[](d);
    // add each box as different geometry to the scene (will be able to obtain the geometryID and hence the object id)
    AddArbitraryBBoxToScene(*structure, pvol, fBuildMode);
  }

  // commit the scene
  rtcCommitScene(scene);

  auto elapsed = timer.Stop();
  std::cerr << "EMBREE SETUP TOOK " << elapsed << "s \n";
  return structure;
}

EmbreeManager::EmbreeAccelerationStructure *EmbreeManager::BuildStructureFromBoundingBoxes(
    ABBoxManager<Precision>::ABBoxContainer_t abboxes, size_t numberofdaughters) const
{
  Stopwatch timer;
  timer.Start();
  // init (device) + scene
  // make the scene
  auto device = rtcNewDevice("VecGeomDevice"); // --> could be a global device??
  auto scene  = rtcNewScene(device);
  rtcSetSceneFlags(scene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
  rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

  auto structure     = new EmbreeManager::EmbreeAccelerationStructure();
  structure->fDevice = device;
  structure->fScene  = scene;

  structure->fNormals       = new Vector3D<float>[numberofdaughters * 6];
  structure->fNumberObjects = numberofdaughters;

  for (size_t d = 0; d < numberofdaughters; ++d) {
    auto lower = abboxes[2 * d];
    auto upper = abboxes[2 * d + 1];
    // add each box as different geometry to the scene (will be able to obtain the geometryID and hence the object id)
    AddBoxGeometryToScene(*structure, lower, upper);
  }

  // commit the scene
  rtcCommitScene(scene);

  auto elapsed = timer.Stop();
  std::cerr << "EMBREE SETUP TOOK " << elapsed << "s \n";
  return structure;
}

void EmbreeManager::RemoveStructure(LogicalVolume const *lvol)
{
  // FIXME: take care of memory deletion within acceleration structure
  if (fStructureHolder[lvol->id()]) delete fStructureHolder[lvol->id()];
}

// it does not have to be a box? could by any polygonal hull
void EmbreeManager::AddArbitraryBBoxToScene(EmbreeAccelerationStructure &structure, VPlacedVolume const *pvol,
                                            EmbreeBuildMode mode) const
{
  if (!pvol) {
    return;
  }
  const auto transf   = pvol->GetTransformation();
  const auto unplaced = pvol->GetLogicalVolume()->GetUnplacedVolume();

  // calc extend
  Vector3D<double> lower;
  Vector3D<double> upper;
  unplaced->Extent(lower, upper);
  if (mode == EmbreeBuildMode::kAABBox) {
    AddBoxGeometryToScene(structure, lower, upper);
  } else {
    AddBoxGeometryToScene(structure, lower, upper, *transf);
  }
}

#ifdef USETRIANGLES
void EmbreeManager::AddBoxGeometryToScene(EmbreeAccelerationStructure &structure,
                                          Vector3D<Precision> const &lower_local,
                                          Vector3D<Precision> const &upper_local, Transformation3D const &transf) const
{
  auto embreeDevice = structure.fDevice;
  auto embreeScene  = structure.fScene;

  //     6 --- 7
  //    /|    /|
  //  2--4--3  5             y      z
  //  | /   | /              |     /
  //  0 --- 1       ---> x   |    /

  // lower_local is 0
  // upper_local is 7
  // the normals are supposed to face outwards!

  // calculate the individual corners
  const auto dx = Vector3D<Precision>(upper_local.x() - lower_local.x(), 0, 0);
  const auto dy = Vector3D<Precision>(0, upper_local.y() - lower_local.y(), 0);
  const auto dz = Vector3D<Precision>(0, 0, upper_local.z() - lower_local.z());

  const auto c0 = transf.InverseTransform(lower_local);
  const auto c1 = transf.InverseTransform(lower_local + dx);
  const auto c2 = transf.InverseTransform(lower_local + dy);
  const auto c3 = transf.InverseTransform(lower_local + dx + dy);
  const auto c4 = transf.InverseTransform(lower_local + dz);
  const auto c5 = transf.InverseTransform(lower_local + dz + dx);
  const auto c6 = transf.InverseTransform(lower_local + dz + dy);
  const auto c7 = transf.InverseTransform(upper_local);

  /* create a triangulated cube with 12 triangles and 8 vertices */
  unsigned int meshID;
  RTCGeometry geom_0 = rtcNewGeometry(embreeDevice, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryBuildQuality(geom_0, RTC_BUILD_QUALITY_HIGH);
  rtcSetGeometryTimeStepCount(geom_0, 1);
  meshID = rtcAttachGeometry(embreeScene, geom_0);

  // to get vertices in the frame of the scene:
  const auto lower = transf.InverseTransform(lower_local);
  const auto upper = transf.InverseTransform(upper_local);

  struct Vertex {
    float x, y, z, r;
  };
  struct Triangle {
    int v0, v1, v2;
  };

  //
  /* set vertices and vertex colors */
  Vertex *vertices =
      (Vertex *)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 4 * sizeof(float), 8);
  vertices[0].x = c0.x();
  vertices[0].y = c0.y();
  vertices[0].z = c0.z();
  vertices[1].x = c1.x();
  vertices[1].y = c1.y();
  vertices[1].z = c1.z();
  vertices[2].x = c2.x();
  vertices[2].y = c2.y();
  vertices[2].z = c2.z();
  vertices[3].x = c3.x();
  vertices[3].y = c3.y();
  vertices[3].z = c3.z();
  vertices[4].x = c4.x();
  vertices[4].y = c4.y();
  vertices[4].z = c4.z();
  vertices[5].x = c5.x();
  vertices[5].y = c5.y();
  vertices[5].z = c5.z();
  vertices[6].x = c6.x();
  vertices[6].y = c6.y();
  vertices[6].z = c6.z();
  vertices[7].x = c7.x();
  vertices[7].y = c7.y();
  vertices[7].z = c7.z();

  //  /* set triangles and face colors */
  int tri = 0;
  Triangle *triangles =
      (Triangle *)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(int), 12);

  // build up triangles by using indices to vertices

  //     6 --- 7
  //    /|    /|
  //  2--4--3  5             y      z
  //  | /   | /              |     /
  //  0 --- 1       ---> x   |    /

  // lower_local is 0
  // upper_local is 7

  // "front" side
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 1;
  tri++;
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 3;
  tri++;
  // "back" side
  triangles[tri].v0 = 5;
  triangles[tri].v1 = 7;
  triangles[tri].v2 = 6;
  tri++;
  triangles[tri].v0 = 4;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 6;
  tri++;
  // "left" side
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 2;
  tri++;
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 6;
  tri++;
  // "right" side
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 5;
  tri++;
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 7;
  triangles[tri].v2 = 5;
  tri++;
  // "top" side
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 6;
  tri++;
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 7;
  tri++;
  // "bottom" side
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 1;
  triangles[tri].v2 = 5;
  tri++;
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 4;
  tri++;

  // calculate normals to detect on which side of triangle we are
  auto calcNormal = [triangles, vertices](int i) {
    const auto &vertex0 = vertices[triangles[i].v0];
    Vector3D<float> p0(vertex0.x, vertex0.y, vertex0.z);
    const auto &vertex1 = vertices[triangles[i].v1];
    Vector3D<float> p1(vertex1.x, vertex1.y, vertex1.z);
    const auto &vertex2 = vertices[triangles[i].v2];
    Vector3D<float> p2(vertex2.x, vertex2.y, vertex2.z);
    // the normal is (p1 - p0) x (p2 - p0);
    Vector3D<float> n = (p1 - p0).Cross(p2 - p0).Normalized().FixZeroes();
    return n;
  };

  // show all normals
  // std::cerr << "-----\n";
  for (int i = 0; i < 12; ++i) {
    // std::cerr << "normal " << i << " " << calcNormal(i) << "\n";
    structure.fNormals[meshID * 12 + i] = calcNormal(i);
    VECGEOM_ASSERT(structure.fNormals[meshID * 12 + i].Mag2() > 0.);
  }

  rtcCommitGeometry(geom_0);
  rtcReleaseGeometry(geom_0);
}
#endif

#define USEQUADS 1

#ifdef USEQUADS
void EmbreeManager::AddBoxGeometryToScene(EmbreeAccelerationStructure &structure,
                                          Vector3D<Precision> const &lower_local,
                                          Vector3D<Precision> const &upper_local, Transformation3D const &transf) const
{
  auto embreeDevice = structure.fDevice;
  auto embreeScene  = structure.fScene;

  //     6 --- 7
  //    /|    /|
  //  2--4--3  5             y      z
  //  | /   | /              |     /
  //  0 --- 1       ---> x   |    /

  // lower_local is 0
  // upper_local is 7
  // the normals are supposed to face outwards!

  // calculate the individual corners
  const auto dx = Vector3D<Precision>(upper_local.x() - lower_local.x(), 0, 0);
  const auto dy = Vector3D<Precision>(0, upper_local.y() - lower_local.y(), 0);
  const auto dz = Vector3D<Precision>(0, 0, upper_local.z() - lower_local.z());

  const auto c0 = transf.InverseTransform(lower_local);
  const auto c1 = transf.InverseTransform(lower_local + dx);
  const auto c2 = transf.InverseTransform(lower_local + dy);
  const auto c3 = transf.InverseTransform(lower_local + dx + dy);
  const auto c4 = transf.InverseTransform(lower_local + dz);
  const auto c5 = transf.InverseTransform(lower_local + dz + dx);
  const auto c6 = transf.InverseTransform(lower_local + dz + dy);
  const auto c7 = transf.InverseTransform(upper_local);

  /* create a triangulated cube with 12 triangles and 8 vertices */
  unsigned int meshID;
  RTCGeometry geom_0 = rtcNewGeometry(embreeDevice, RTC_GEOMETRY_TYPE_QUAD);
  rtcSetGeometryBuildQuality(geom_0, RTC_BUILD_QUALITY_HIGH);
  rtcSetGeometryTimeStepCount(geom_0, 1);
  meshID = rtcAttachGeometry(embreeScene, geom_0);

  // to get vertices in the frame of the scene:
  const auto lower = transf.InverseTransform(lower_local);
  const auto upper = transf.InverseTransform(upper_local);

  struct Vertex {
    float x, y, z, r;
  };
  struct Triangle {
    int v0, v1, v2;
  };
  struct Quad {
    int v0, v1, v2, v3;
  };

  //
  /* set vertices and vertex colors */
  Vertex *vertices =
      (Vertex *)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 4 * sizeof(float), 8);
  vertices[0].x = c0.x();
  vertices[0].y = c0.y();
  vertices[0].z = c0.z();
  vertices[1].x = c1.x();
  vertices[1].y = c1.y();
  vertices[1].z = c1.z();
  vertices[2].x = c2.x();
  vertices[2].y = c2.y();
  vertices[2].z = c2.z();
  vertices[3].x = c3.x();
  vertices[3].y = c3.y();
  vertices[3].z = c3.z();
  vertices[4].x = c4.x();
  vertices[4].y = c4.y();
  vertices[4].z = c4.z();
  vertices[5].x = c5.x();
  vertices[5].y = c5.y();
  vertices[5].z = c5.z();
  vertices[6].x = c6.x();
  vertices[6].y = c6.y();
  vertices[6].z = c6.z();
  vertices[7].x = c7.x();
  vertices[7].y = c7.y();
  vertices[7].z = c7.z();

  //  /* set quads and face colors */
  int tri     = 0;
  Quad *quads = (Quad *)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, 4 * sizeof(int), 6);

  // build up quads by using indices to vertices

  //     6 --- 7
  //    /|    /|
  //  2--4--3  5             y      z
  //  | /   | /              |     /
  //  0 --- 1       ---> x   |    /

  // lower_local is 0
  // upper_local is 7

  // "front" side
  quads[tri].v0 = 0;
  quads[tri].v1 = 2;
  quads[tri].v2 = 3;
  quads[tri].v3 = 1;

  tri++;
  // "back" side
  quads[tri].v0 = 5;
  quads[tri].v1 = 7;
  quads[tri].v2 = 6;
  quads[tri].v3 = 4;

  tri++;
  // "left" side
  quads[tri].v0 = 0;
  quads[tri].v1 = 4;
  quads[tri].v2 = 6;
  quads[tri].v3 = 2;

  tri++;
  // "right" side
  quads[tri].v0 = 1;
  quads[tri].v1 = 3;
  quads[tri].v2 = 7;
  quads[tri].v3 = 5;

  tri++;
  // "top" side
  quads[tri].v0 = 3;
  quads[tri].v1 = 2;
  quads[tri].v2 = 6;
  quads[tri].v3 = 7;

  tri++;
  // "bottom" side
  quads[tri].v0 = 0;
  quads[tri].v1 = 1;
  quads[tri].v2 = 5;
  quads[tri].v3 = 4;

  // calculate normals to detect on which side of triangle we are
  auto calcNormal = [quads, vertices](int i) {
    const auto &vertex0 = vertices[quads[i].v0];
    Vector3D<float> p0(vertex0.x, vertex0.y, vertex0.z);
    const auto &vertex1 = vertices[quads[i].v1];
    Vector3D<float> p1(vertex1.x, vertex1.y, vertex1.z);
    const auto &vertex2 = vertices[quads[i].v2];
    Vector3D<float> p2(vertex2.x, vertex2.y, vertex2.z);
    // the normal is (p1 - p0) x (p2 - p0);
    Vector3D<float> n = (p1 - p0).Cross(p2 - p0).Normalized().FixZeroes();
    return n;
  };

  // show all normals
  // std::cerr << "-----\n";
  for (int i = 0; i < 6; ++i) {
    // std::cerr << "normal " << i << " " << calcNormal(i) << "\n";
    structure.fNormals[meshID * 6 + i] = calcNormal(i);
    VECGEOM_ASSERT(structure.fNormals[meshID * 6 + i].Mag2() > 0.);
  }

  rtcCommitGeometry(geom_0);
  rtcReleaseGeometry(geom_0);
}
#endif

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
