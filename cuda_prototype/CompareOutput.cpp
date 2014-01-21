#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

int main(const int argc, const char * const argv[]) {
  if (argc < 2) {
    std::cout << "Please specify the input file to compare.\n";
    return -1;
  }
  std::string filename(argv[1]);
  std::ifstream file_cuda, file_vc;
  file_cuda.open(filename + ".out.cuda");
  file_vc.open(filename + ".out.vc");
  int count = 0;
  int equal = 0;
  while (file_cuda && file_vc) {
    count++;
    std::string val_cuda, val_vc;
    std::getline(file_cuda, val_cuda);
    std::getline(file_vc, val_vc);
    const double cuda = std::atof(val_cuda.c_str());
    const double vc = std::atof(val_vc.c_str());
    if (std::fabs(cuda - vc) < 1e-12 || cuda == vc) {
      equal++;
    } else {
      // std::cout << "Mismatch: " << vc << " / " << cuda << std::endl;
    }
  }
  std::cout << (count - equal) << " / " << count << " mismatches detected.\n";
  return 0;
}