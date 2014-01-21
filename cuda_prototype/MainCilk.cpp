#include <iostream> 
#include "LibraryCilk.h"
#include "Box.h"

double random(const double low, const double high) {
  return low + static_cast<double>(rand()) /
               static_cast<double>(RAND_MAX / (high - low));
}

int main(void) {
  return 0;
}