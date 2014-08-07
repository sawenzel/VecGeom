#include "base/SOA.h"

using namespace vecgeom;

int main() {
  SOA<int, 6, 4> data;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 6; ++j) {
      data[i][j] = 10*i + j;
      std::cout << (i*6+j) << ": " << data[i][j] << "\n";
    }
  }
  return 0;
}