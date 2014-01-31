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
  std::ifstream file_cuda, file_vc, file_cilk;
  file_cuda.open(filename + ".out.cuda");
  file_vc.open(filename + ".out.vc");
  file_cilk.open(filename + ".out.cilk");
  int count = 0;
  int equal = 0;
  while (file_cuda && file_vc && file_cilk) {
    count++;
    std::string val_cuda, val_vc, val_cilk;
    std::getline(file_cuda, val_cuda);
    std::getline(file_vc, val_vc);
    std::getline(file_cilk, val_cilk);
    const double cuda = std::atof(val_cuda.c_str());
    const double vc = std::atof(val_vc.c_str());
    const double cilk = std::atof(val_cilk.c_str());
    if ((std::fabs(cuda - vc) < 1e-12 || cuda == vc) &&
        (std::fabs(vc - cilk) < 1e-12 || vc == cilk)) {
      equal++;
    } else {
      std::cout << "Mismatch: " << vc << " / " << cuda << " / "
                << cilk << std::endl;
    }
  }
  std::cout << (count - equal) << " / " << count << " mismatches detected.\n";
  return 0;
}