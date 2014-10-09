#include "management/Visualizer.h"
#include "volumes/Tube.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 512;
  SimpleTube tube("Visualizer Tube", 2, 4, 4, kPi/4, kPi/4);
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(4, 4, 4));
    } while (!tube.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer::Instance().AddVolume(tube);
  Visualizer::Instance().AddPoints(points);
  Visualizer::Instance().Show();
  return 0;
}