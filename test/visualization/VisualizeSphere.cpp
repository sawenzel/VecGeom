#include "utilities/Visualizer.h"
#include "volumes/Sphere.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 512;
  SimpleSphere sphere("Visualizer Sphere", 4, 5, 0.0, 2*kPi, kPi/6 , kPi/6);
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!sphere.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(sphere);
  //visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}