#include "volumes/Planes.h"
#include "backend/Backend.h"

namespace VECGEOM_NAMESPACE {

Planes::Planes(int size) : fNormals(size), fDistances(size) {}

Planes::~Planes() {}

Planes& Planes::operator=(Planes const &other) {
  fNormals = other.fNormals;
  fDistances = other.fDistances;
  return *this;
}

void Planes::Set(
    int index,
    Vector3D<Precision> const &normal,
    Vector3D<Precision> const &x0) {
  Vector3D<Precision> fixedNormal(normal);
  fixedNormal.FixZeroes();
  Precision inverseLength = 1. / fixedNormal.Mag();
  fNormals.set(index, inverseLength*fixedNormal);
  fDistances[index] = inverseLength * -fixedNormal.Dot(x0);
}

void Planes::Set(
    int index,
    Vector3D<Precision> const &normal,
    Precision distance) {
  fNormals.set(index, normal);
  fDistances[index] = distance;
}

void Planes::FlipSign(int index) {
  fNormals.set(index, -fNormals[index]);
  fDistances[index] = -fDistances[index];
}

std::ostream& operator<<(std::ostream &os, Planes const &planes) {
  for (int i = 0, iMax = planes.size(); i < iMax; ++i) {
    os << "{" << planes.GetNormal(i) << ", " << planes.GetDistance(i) << "}\n";
  }
  return os;
}

}