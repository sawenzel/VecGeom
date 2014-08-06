#include "base/SOA.h"

using namespace vecgeom;

int main() {
  SOAData<double, 6, 4>* data = new SOAData<double, 6, 4>;
  std::cout << "Address: " << data << "\n"
            << "Size: " << sizeof(*data) << "\n"
            << "Address of next: " << &data->fTail << "\n"
            << "Difference in address: "
            << ((unsigned long)&data->fTail - (unsigned long)data)
            << "\n"
            << "Address of next: " << &data->fTail.fTail << "\n"
            << "Difference in address: "
            << ((unsigned long)&data->fTail.fTail - (unsigned long)&data->fTail)
            << "\n";
  return 0;
}