#include <iostream>
#include <stdio.h>
#include <cassert>

__global__
void MyAbs() {
  printf("Hello the world");
  float a = -0.38;
  float b = abs(a);
  printf("%f -> %f\n", a, b);
}

__host__
inline __attribute__((always_inline))
void CheckCudaError() {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA reported error with message: \""
              << cudaGetErrorString(err) << "\"\n";
    assert(err != cudaSuccess);
  }
}

__host__
int main(void) {
  MyAbs<<<8,1>>>();
  CheckCudaError();
  return 0;
}