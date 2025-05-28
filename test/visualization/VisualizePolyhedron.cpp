#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 100000;

  Precision phiStart = kPi / 4., deltaPhi = kPi / 3. + kPi / 4.;
  constexpr int sides   = 8;
  constexpr int nPlanes = 6;

  Precision rmin[] = {0., 0., 3., 3., 0., 0.};
  Precision rmax[] = {3.5, 3.5, 3.5, 3.5, 3.5, 3.5};
  Precision z[]    = {-3., -2.5, -2.5, 2.5, 2.5, 3.};

  // Precision rmin[] = {0.,0.,0.,0.};
  // Precision rmax[] = {15.,15.,15.,10.};
  // Precision z[]    = {0.,20.,30.,40.};

  // Precision rmin[] = {0.,0.,0.,0.,0.,0.};
  // Precision rmax[] = {10.,20.,20.,10.,10.,5.};
  // Precision z[]    = {0.,20.,20.,40.,40.,50.};

  // Precision z[]    = {0.,20.,20.,40.,45.,50.,50.,60.};
  // Precision rmin[] = {0.,0.,0.,0.,0.,0.,0.,0.};
  // Precision rmax[] = {10.,20.,20.,10.,10.,5.,5.,20.};

  // Precision z[]    = {-2, -1, 1, 2};
  // Precision rmin[] = {0, 1, 0.5, 0};
  // Precision rmax[] = {1, 2, 2, 1};

  SimplePolyhedron polyhedron("Visualizer Polyhedron", phiStart, deltaPhi, sides, nPlanes, z, rmin, rmax);
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);

  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    sample = polyhedron.GetUnplacedVolume()
                 ->SamplePointOnSurface(); // volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    VECGEOM_ASSERT(polyhedron.Inside(sample) == vecgeom::EInside::kSurface);
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }

  Visualizer visualizer;
  visualizer.AddVolume(polyhedron);
  visualizer.AddPoints(pm);
  visualizer.Show();
  return 0;
}
