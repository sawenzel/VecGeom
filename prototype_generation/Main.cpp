#include <stdio.h>
#include "PlacedBox.h"
// #include "SpecializedTube.h"

int main() {
  UnplacedBox unplaced_box(5., 5., 5.);
  // UnplacedTube unplaced_tube();
  PlacedBox placed_box(&unplaced_box);
  // SpecializedTube<GeneralTube> general_tube(&unplaced_tube);
  // SpecializedTube<FancyTube> fancy_tube(&unplaced_tube);

  const double point[3] = {1, -4, 0};
  double points[3][VcDouble::Size] __attribute__((aligned(32)));
  bool output[VcDouble::Size] __attribute__((aligned(32)));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < VcDouble::Size; ++j) points[i][j] = point[i];
  }

  placed_box.Inside(points, output);
  printf("Box: ");
  for (int i = 0; i < VcDouble::Size; ++i) printf("%i ", output[i]);

  // general_tube.template Inside(points, output);
  // printf("\nGeneral Tube: ");
  // for (int i = 0; i < VcDouble::Size; ++i) printf("%i ", output[i]);

  // fancy_tube.Inside(points, output);
  // printf("\nFancy Tube: ");
  // for (int i = 0; i < VcDouble::Size; ++i) printf("%i ", output[i]);

  printf("\n");

  return 0;
}