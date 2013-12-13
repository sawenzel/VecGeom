#include <iostream>
#include "ShapeTester.h"

const std::vector<std::vector<double>> rot_cases {
  { 0,  0,   0},
  {30,  0,   0},
  { 0, 45,   0},
  { 0,  0, 675},
  { 0,  0, 180},
  {30, 48,   0},
  {78, 81, -10}
};
const std::vector<std::vector<double>> trans_cases {
  {0,     0,   0},
  {10,    0,   0},
  {10,  -10,   0},
  {10, -100, 100}
};

int main(void) {

  ShapeTester *tester = new ShapeTester();
  tester->SetRepetitions(1024);
  tester->SetVerbose(true);

  TransformationMatrix const *identity =
      new TransformationMatrix(0, 0, 0, 0, 0, 0);
  BoxParameters const *world_params = new BoxParameters(100, 100, 100);

  const double rmin = 10;
  const double rmax = 20;
  const double dz = 30;
  const double phis = 0;
  const double dphi = M_PI;
  TubeParameters<double> const *tube_params =
      new TubeParameters<double>(rmin, rmax, dz, phis, dphi);

  const int n_rotcases = rot_cases.size();
  const int n_transcases = trans_cases.size();
  const int n_totalcases = n_rotcases * n_transcases;
      
  // Initialize empty world
  PhysicalVolume *world = GeoManager::MakePlacedBox(world_params, identity);
  tester->SetWorld(world);

  for (int r = 0; r < n_rotcases; ++r) {
    for (int t = 0; t < n_transcases; ++t) {

      // Generate specialized transformation matrix for case
      TransformationMatrix const *sm =
          TransformationMatrix::createSpecializedMatrix(
              trans_cases[t][0],  trans_cases[t][1],  trans_cases[t][2],
              rot_cases[r][0],    rot_cases[r][1],    rot_cases[r][2]);
      PhysicalVolume const *tube = GeoManager::MakePlacedTube(tube_params, sm);
      world->AddDaughter(tube);

      // Run benchmark and do cleanup
      // tester->Run();
      // delete sm;
      // delete tube;
      // delete world;
    }
  }
  tester->Run();

  std::vector<ShapeBenchmark> results = tester->Results();
  // for (int i = 0; i < results.size(); ++i) results[i].Print();

  // Clean
  delete identity;
  delete world_params;
  delete tube_params;
  delete tester;

  return 0;
}