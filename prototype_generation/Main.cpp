#include <stdio.h>
#include "SimpleBox.h"
#include "SimpleTube.h"
#include "Vector3D.h"

int main() {

  const int n = 8;

  UnplacedBox unplacedBox = UnplacedBox(5., 5., 5.);
  UnplacedTube unplacedTube = UnplacedTube();
  SimpleBox simpleBox = SimpleBox(&unplacedBox);
  SimpleTube simpleTube(&unplacedTube);
  SpecializedTube<FancyTube> fancyTube(&unplacedTube);

  const Vector3D<double> point(-1., 2., 0);
  double **const points = new double*[3];
  points[0] = new double[n];
  points[1] = new double[n];
  points[2] = new double[n];
  bool *const output = new bool[n];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < n; ++j) points[i][j] = point[i];
  }

  simpleBox.Inside(points, n, output);
  printf("Box: ");
  for (int i = 0; i < n; ++i) printf("%i ", output[i]);

  simpleTube.Inside(points, n, output);
  printf("\nSimple Tube: ");
  for (int i = 0; i < n; ++i) printf("%i ", output[i]);

  fancyTube.Inside(points, n, output);
  printf("\nFancy Tube: ");
  for (int i = 0; i < n; ++i) printf("%i ", output[i]);

  printf("\n");

  return 0;
}