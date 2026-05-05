#include "VecGeom/base/Config.h"

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeomTest/Benchmarker.h"
#include "VecGeom/management/GeoManager.h"
#include "ArgParser.h"
#include "VecGeom/base/Stopwatch.h"
#include <iostream>
#include "VecGeom/volumes/Tessellated.h"
#include "VecGeom/volumes/Box.h"
#include "test/core/TessellatedOrb.h"
#include <string>

#ifdef VECGEOM_EMBREE
#include <embree3/rtcore.h>
#endif

using namespace vecgeom;

#ifdef VECGEOM_EMBREE
//> An meshed shape which uses the Embree data structures and layout
class EmbreeMeshShape {
public:
  EmbreeMeshShape(UnplacedTessellated const &);
  void InitMeshFromTesselatedSolid(UnplacedTessellated const &);

  virtual double DistanceToIn(double x, double y, double z, double dx, double dy, double dz)
  {
    RTCRayHit ray;
    ray.ray.flags = 0;
    ray.ray.org_x = x;
    ray.ray.org_y = y;
    ray.ray.org_z = z;
    ray.ray.dir_x = dx;
    ray.ray.dir_y = dy;
    ray.ray.dir_z = dz;
    ray.ray.tnear = 0.;
    ray.ray.tfar  = 1E20f;
    {
      RTCIntersectContext context;
      rtcInitIntersectContext(&context);
      rtcIntersect1(fScene, &context, &ray);
      ray.hit.Ng_x = -ray.hit.Ng_x; // EMBREE_FIXME: only correct for triangles,quads, and subdivision surfaces
      ray.hit.Ng_y = -ray.hit.Ng_y;
      ray.hit.Ng_z = -ray.hit.Ng_z;
    }
    return ray.ray.tfar;
  }

private:
  RTCDevice fDevice = nullptr;
  RTCScene fScene   = nullptr;
};

EmbreeMeshShape::EmbreeMeshShape(const UnplacedTessellated &tsl)
{
  InitMeshFromTesselatedSolid(tsl);
}

void EmbreeMeshShape::InitMeshFromTesselatedSolid(const UnplacedTessellated &tsl)
{
  fDevice = rtcNewDevice("VecGeomDevice"); // --> could be a global device??
  fScene  = rtcNewScene(fDevice);
  rtcSetSceneBuildQuality(fScene, RTC_BUILD_QUALITY_HIGH);

  unsigned int meshID;
  RTCGeometry geom_0 = rtcNewGeometry(fDevice, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryBuildQuality(geom_0, RTC_BUILD_QUALITY_HIGH);
  rtcSetGeometryTimeStepCount(geom_0, 1);
  meshID = rtcAttachGeometry(fScene, geom_0);

  struct Vertex {
    float x, y, z, r;
  };
  struct Triangle {
    int v0, v1, v2;
  };

  // determine number of vertices and triangles
  int ntriangles = tsl.GetNFacets();
  int nvertices  = 3 * ntriangles; // lets assume they are all different so that we don't have to match the vertices

  //
  /* set vertices and vertex colors */
  Vertex *vertices    = (Vertex *)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                       4 * sizeof(float), nvertices);
  Triangle *triangles = (Triangle *)rtcSetNewGeometryBuffer(geom_0, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                            3 * sizeof(int), ntriangles);

  // loop over all tessellated triangles and add them to the embree buffer
  int vertexcounter = 0;
  for (int t = 0; t < ntriangles; ++t) {
    auto tile = tsl.GetFacet(t);
    auto v    = tile->fVertices;
    // tile->fIndices;

    vertices[vertexcounter].x = v[0].x();
    vertices[vertexcounter].y = v[0].y();
    vertices[vertexcounter].z = v[0].z();
    triangles[t].v0           = vertexcounter;
    vertexcounter++;

    vertices[vertexcounter].x = v[1].x();
    vertices[vertexcounter].y = v[1].y();
    vertices[vertexcounter].z = v[1].z();
    triangles[t].v1           = vertexcounter;
    vertexcounter++;

    vertices[vertexcounter].x = v[2].x();
    vertices[vertexcounter].y = v[2].y();
    vertices[vertexcounter].z = v[2].z();
    triangles[t].v2           = vertexcounter;
    vertexcounter++;
  }

  rtcCommitGeometry(geom_0);
  rtcReleaseGeometry(geom_0);
  rtcCommitScene(fScene);
}
#endif // VECGEOM_EMBREE

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 1024);
  OPTION_INT(nrep, 4);
  OPTION_DOUBLE(ngrid, 100);
  OPTION_STRING(obj, {""}); // tesselated from object file
  OPTION_INT(verbose, 1);
  double r = 10. * ngrid;

  UnplacedBox worldUnplaced = UnplacedBox(2. * r, 2. * r, 2. * r);

  UnplacedTessellated tsl;
  size_t nfacets{0};
  if (obj.size() > 0) {
    // load object to benchmark from .obj file
    nfacets = tsl.FillFromObjFile(obj);
  } else {
    // Create the tessellated solid
    nfacets = TessellatedOrb(r, ngrid, tsl);
    tsl.Close();
  }

  Stopwatch timer;
  timer.Start();

  auto elapsed = timer.Stop();
  std::cout << "SETUP TOOK " << elapsed << " s\n";

  std::cout << "Benchmarking tessellated solid having " << nfacets << " facets\n";

  LogicalVolume world("world", &worldUnplaced);
  LogicalVolume tessellated("tessellated", &tsl);

  Transformation3D placement(0, 0, 0);
  const VPlacedVolume *placedTsl = world.PlaceDaughter("tessellated", &tessellated, &placement);
  (void)placedTsl;

  VPlacedVolume *worldPlaced = world.Place();

  GeoManager::Instance().SetWorldAndClose(worldPlaced);

  Benchmarker tester(GeoManager::Instance().GetWorld());
  tester.SetVerbosity(verbose);
  tester.SetRepetitions(nrep);
  tester.SetPointCount(npoints);
  tester.SetToInBias(0.8);
  tester.SetPoolMultiplier(1);
  tester.SetTolerance(1E-4);

  // tester.RunToInBenchmark();
  tester.RunBenchmark();

  auto &rays = tester.GetProblematicRays();
  std::cerr << "have " << rays.size() << "rays\n";

#ifdef VECGEOM_EMBREE
  // this part enables comparison to a pure EMBREE represented tesselated solid
  // just to see what is maximally possible if we used floating point precision and no error handling
  timer.Start();
  EmbreeMeshShape mesh(tsl);
  elapsed = timer.Stop();
  std::cout << "EMBREE SETUP TOOK " << elapsed << " s\n";

  auto &pointPool = tester.GetPointPool();
  auto &dirPool   = tester.GetDirectionPool();

  // Stopwatch timer;
  timer.Start();
  double s = 0;
  for (int i = 0; i < pointPool.size(); ++i) {
    const auto &p = pointPool[i];
    const auto &d = dirPool[i];
    auto dist     = mesh.DistanceToIn(p[0], p[1], p[2], d[0], d[1], d[2]);

    // std::cerr << "EMBREE " << dist << "\n";
    s += dist;
  }
  elapsed = timer.Stop();
  std::cerr << "time took " << elapsed << " s "
            << " CHECKSUM " << s << "\n";
#endif
  //  tester.RunToOutBenchmark();
  return 0; // tester.RunBenchmark();
}
