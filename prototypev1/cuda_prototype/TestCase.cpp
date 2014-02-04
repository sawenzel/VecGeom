#include <iostream>
#include <string>
#include <fstream>
#include "LibraryGeneric.h"
#include "Box.h"

int main(void) {
  
  const int n_points = 1<<21;

  const std::string filename("io/single_box.in");

  const TransMatrix<double> *origin = new TransMatrix<double>();
  TransMatrix<double> *pos = new TransMatrix<double>();
  pos->SetTranslation(2.93, 1.30, -4.05);
  Box world(new BoxParameters(Vector3D<double>(10., 10., 10.)), origin);
  Box box(new BoxParameters(Vector3D<double>(2., 1., 4.)), pos);
  world.AddDaughter(&box);

  SOA3D<double> points(n_points);
  SOA3D<double> directions(n_points);
  std::cerr << "Filling uncontained points...";
  world.FillUncontainedPoints(points);
  std::cerr << " Done. \nFilling biased directions...";
  world.FillBiasedDirections(points, 0.666, directions);
  std::cerr << " Done.\n";

  std::ofstream filestream;
  filestream.open(filename);
  std::cerr << "Writing to file \"" << filename << "\"...\n";
  for (int i = 0; i < n_points; ++i) {
    filestream << points[i] << ";" << directions[i] << std::endl;
  }
  filestream.close();

  return 0;
}