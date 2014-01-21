#include <iostream> 
#include <fstream>
#include "LibraryVc.h"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
}

int main(void) {

  Stopwatch timer, timer_total;
  timer_total.Start();
  
  const int n_points = 1<<21;

  const TransMatrix<double> *origin = new TransMatrix<double>();
  TransMatrix<double> *pos = new TransMatrix<double>();
  pos->SetTranslation(2.93, 1.30, -4.05);
  Box world(Vector3D<double>(10., 10., 10.), origin);
  Box box(Vector3D<double>(2., 1., 4.), pos);
  world.AddDaughter(&box);

  SOA3D<double> points(n_points);
  SOA3D<double> directions(n_points);
  timer.Start();
  std::cerr << "Loading points and directions...";
  std::string filename("io/single_box.in");
  std::ifstream filestream;
  filestream.open(filename);
  for (int i = 0; i < n_points; ++i) {
    std::string line;
    std::getline(filestream, line);
    const int semicolon = line.find(";");
    points.Set(i, Vector3D<double>(line.substr(0, semicolon)));
    directions.Set(i, Vector3D<double>(line.substr(semicolon+1)));
  }
  filestream.close();
  std::cerr << " Done in " << timer.Stop() << "s.\n";

  double *step_max = (double*) _mm_malloc(sizeof(double)*n_points,
                                          kAlignmentBoundary);
  double *output = (double*) _mm_malloc(sizeof(double)*n_points,
                                        kAlignmentBoundary);
  timer.Start();
  for (int i = 0; i < n_points; ++i) {
    step_max[i] = kInfinity;
  }
  std::cout << "Max step array initialized in " << timer.Stop() << "s.\n";

  timer.Start();
  box.DistanceToIn(points, directions, step_max, output);
  
  std::cout << "Vc benchmark for " << n_points << " points finished in "
            << timer.Stop() << "s.\n";

  int hit = 0;
  filename = "io/single_box.out.vc";
  std::ofstream outstream;
  outstream.open(filename);
  for (int i = 0; i < n_points; ++i) {
    outstream << output[i] << std::endl;
    if (output[i] < kInfinity) {
      hit++;
    }
  }
  outstream.close();
  std::cout << "Hits counted in " << timer.Stop() << "s.\n";

  std::cout << hit << " / " << n_points << " (" << double(hit)/double(n_points)
            << " of total) hit something.\n";

  std::cout << "Total binary execution time: " << timer_total.Stop() << "s.\n";

  return 0;
}