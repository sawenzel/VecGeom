#include "Shape.h"
#include "Box.h"
#include <vector>
#include <random>

static std::mt19937 rng(1);
std::uniform_real_distribution<> uniform_dist(0,1);

Vector3D<double> Shape::SamplePoint(Vector3D<double> const &size,
                                    const double scale) {
  return Vector3D<double>(
    scale * (1. - 2. * uniform_dist(rng)) * size[0],
    scale * (1. - 2. * uniform_dist(rng)) * size[1],
    scale * (1. - 2. * uniform_dist(rng)) * size[2]
  );
}

Vector3D<double> Shape::SampleDirection() {

  Vector3D<double> dir(
    (1. - 2. * uniform_dist(rng)),
    (1. - 2. * uniform_dist(rng)),
    (1. - 2. * uniform_dist(rng))
  );

  const double inverse_norm =
      1. / sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  dir[0] *= inverse_norm;
  dir[1] *= inverse_norm;
  dir[2] *= inverse_norm;

  return dir;
}

void Shape::FillRandomDirections(SOA3D<double> &dirs) {

  const int size = dirs.size();
  for (int i = 0; i < size; ++i) {
    const Vector3D<double> temp = SampleDirection();
    dirs.x(i) = temp[0];
    dirs.y(i) = temp[1];
    dirs.z(i) = temp[2];
  }

}

template <typename ShapeType>
inline bool IsFacingShape(Vector3D<double> const &point,
                          Vector3D<double> const &dir,
                          ShapeType const * const shape) {
  return shape->DistanceToIn(point, dir) < kInfinity;
}

void Shape::FillBiasedDirections(SOA3D<double> const &points,
                                 const double bias, SOA3D<double> &dirs) const {
  const int size = dirs.size();
  assert(0. <= bias <= 1.);

  // Randomize points
  FillRandomDirections(dirs);

  // Check hits
  int n_hits = 0;
  std::vector<bool> hit(size, false);
  for (int i = 0; i < size; ++i) {
    for (auto j = daughters.begin(); j != daughters.end(); ++j) {
      if (IsFacingShape(points[i], dirs[i], *j)) {
        n_hits++;
        hit[i] = true;
      }
    }
  }

  // Remove hits until threshold
  int h = 0;
  while (n_hits / size > bias) {
    if (!hit[h]) {
      h++;
      continue;
    }
    do {
      dirs.Set(h, SampleDirection());
      for (auto i = daughters.begin(); i != daughters.end(); ++i) {
        if (!IsFacingShape(points[h], dirs[h], *i)) {
          n_hits--;
          hit[h] = false;
          break;
        }
      }
    } while (hit[h]);
    h++;
  }

  // Add hits until threshold
  h = 0;
  while (n_hits / size < bias) {
    if (hit[h]) {
      h++;
      continue;
    }
    do {
      dirs.Set(h, SampleDirection());
      for (auto i = daughters.begin(); i != daughters.end(); ++i) {
        if (IsFacingShape(points[h], dirs[h], *i)) {
          n_hits++;
          hit[h] = true;
          break;
        }
      }
    } while (!hit[h]);
    h++;
  }

}

void Shape::FillUncontainedPoints(SOA3D<double> &points) const {
  const int size = points.size();
  for (int i = 0; i < size; i++) {
    bool contained = false;
    do {
      points.Set(i, SamplePoint(bounding_box->Dimensions()));
      for (auto j = daughters.begin(); j != daughters.end(); ++j) {
        if ((*j)->Contains(points[i])) {
          contained = true;
          break;
        }
      }
    } while (contained);
  }
}