///
/// \file Utils3D.cpp
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#include "VecGeom/base/Utils3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Utils3D {

using vecCore::math::Abs;
using vecCore::math::Max;

#ifndef VECCORE_CUDA
std::ostream &operator<<(std::ostream &os, Plane const &hpl)
{
  os << "   plane normal: " << hpl.fNorm << "  distance = " << hpl.fDist;
  return os;
}

std::ostream &operator<<(std::ostream &os, Polygon const &poly)
{
  os << "   polygon (";
  for (size_t i = 0; i < poly.fN; ++i)
    os << i << ":" << poly.GetVertex(i) << "  ";
  os << "  normal: " << poly.fNorm << "  distance = " << poly.fDist;
  return os;
}

std::ostream &operator<<(std::ostream &os, Polyhedron const &polyh)
{
  os << "   polyhedron:\n";
  for (size_t i = 0; i < polyh.GetNpolygons(); ++i)
    os << "   " << polyh.GetPolygon(i) << std::endl;
  return os;
}
#endif

void Plane::Transform(Transformation3D const &tr)
{
  // Transform normal vector
  Vec_t tempdir;
  tr.InverseTransformDirection(fNorm, tempdir);
  fNorm = tempdir;
  fDist -= fNorm.Dot(tr.Translation());
}

VECCORE_ATT_HOST_DEVICE
Polygon::Polygon(size_t n, vector_t<Vec_t> &vertices, bool convex)
    : fN(n), fConvex(convex), fNorm(), fVert(&vertices), fInd(n), fSides(n)
{
  VECGEOM_ASSERT(fN > 2);
}

VECCORE_ATT_HOST_DEVICE
Polygon::Polygon(size_t n, vector_t<Vec_t> &vertices, Vec_t const &normal)
    : fN(n), fConvex(true), fHasNorm(true), fNorm(normal), fVert(&vertices), fInd(n), fSides(n)
{
  VECGEOM_ASSERT(fN > 2 && fNorm.IsNormalized());
}

Polygon::Polygon(size_t n, vector_t<Vec_t> &vertices, vector_t<size_t> const &indices, bool convex)
    : fN(n), fConvex(convex), fNorm(), fVert(&vertices), fInd(indices), fSides(n)
{
  CheckAndFixDegenerate();
}

void Polygon::CheckAndFixDegenerate()
{

  if (fValid) {
    return;
  }

  vector_t<size_t> validIndices;
  validIndices.push_back(fInd[0]);
  for (size_t i = 1; i < fN; i++) {
    auto diff1 = (*fVert)[fInd[i]] - (*fVert)[validIndices[0]];
    auto diff2 = (*fVert)[fInd[i]] - (*fVert)[validIndices[validIndices.size() - 1]];

    if (diff1.Mag2() > kTolerance && diff2.Mag2() > kTolerance) {
      validIndices.push_back(fInd[i]);
    }
  }

  fN   = validIndices.size();
  fInd = validIndices;
  fSides.resize(fN, 0);
  if (fN > 2) {
    fValid = true;
  }
}

bool Polygon::isConvexVertex(size_t i0, size_t i1, size_t i2) const
{
  return fNorm.Dot(((*fVert)[i2] - (*fVert)[i1]).Cross((*fVert)[i0] - (*fVert)[i1])) >= 0.;
}

bool Polygon::isPointInsideTriangle(const Vec_t &p, size_t i0, size_t i1, size_t i2) const
{

  Vec_t &A = (*fVert)[i0];
  Vec_t &B = (*fVert)[i1];
  Vec_t &C = (*fVert)[i2];

  Vec_t u = B - A;

  Vec_t v = C - A;

  Vec_t w = p - A;

  Vec_t vCrossW = v.Cross(w);

  Vec_t vCrossU = v.Cross(u);

  if (vCrossW.Dot(vCrossU) < 0) return false;

  Vec_t uCrossW = u.Cross(w);

  Vec_t uCrossV = u.Cross(v);

  if (uCrossW.Dot(uCrossV) < 0) return false;

  double denom = uCrossV.Length();

  double r = vCrossW.Length() / denom;

  double t = uCrossW.Length() / denom;

  // std::cerr << p << ' ' << A << ' ' << B << ' ' << C << ' ' << r << ' ' << t << '\n';

  return (r + t <= 1);
}

void Polygon::TriangulatePolygon(vector_t<Polygon> &polys) const
{

  vector_t<size_t> ind = fInd;

  for (size_t i0 = 0, i1 = 1, i2 = 2; ind.size() > 2;) {

    while (!isConvexVertex(ind[i0], ind[i1], ind[i2]))
      i0++, i1++, i2 = (i2 + 1) % ind.size();

    // (*fVert)[ind[i1]] is a convex vertex
    bool pointInsideTriangle = false;
    for (size_t j = 0; j < ind.size(); j++) {
      if (j != i0 && j != i1 && j != i2 && isPointInsideTriangle((*fVert)[ind[j]], ind[i0], ind[i1], ind[i2])) {
        pointInsideTriangle = true;
        i0++, i1++, i2 = (i2 + 1) % ind.size();
        break;
      }
    }

    if (!pointInsideTriangle) {
      polys.push_back({3, (*fVert), {ind[i0], ind[i1], ind[i2]}, true});
      ind.erase(ind.begin() + i1);
      i0 = 0, i1 = 1, i2 = 2;
    }
  }
}

bool Line::IsPointOnLine(const Vec_t &p)
{
  if (p == fPts[0] || p == fPts[1]) {
    return true;
  }

  Vec_t ap = p - this->fPts[0];
  Vec_t bp = p - this->fPts[1];
  if (ap.Cross(bp).Mag() != 0) {
    return false;
  }

  if (ap.Dot(bp) < 0) {
    return true;
  } else
    return false;
}

LineIntersection *Line::Intersect(const Line &l2)
{
  LineIntersection *li = new LineIntersection();

  const Vec_t &p1 = fPts[0];
  const Vec_t &p2 = fPts[1];
  const Vec_t &p3 = l2.fPts[0];
  const Vec_t &p4 = l2.fPts[1];

  // parallel
  if ((p2 - p1).Cross(p4 - p3).Mag2() == 0.) {
    Vec_t v         = (p2 - p1).Normalized();
    double distance = v.Cross(p3 - p1).Mag();
    // overlapping
    if (distance == 0.) {
      Vec_t v1  = (p2 - p1);
      li->fA    = (p3 - p1).Dot(v1) / (v1.Dot(v1));
      li->fB    = (p4 - p1).Dot(v1) / (v1.Dot(v1));
      li->fType = LineIntersection::fOverlap;
    } else {
      li->fType = LineIntersection::fParallel;
    }
  } else {
    double d1343 = (p1 - p3).Dot(p4 - p3);
    double d4321 = (p4 - p3).Dot(p2 - p1);
    double d1321 = (p1 - p3).Dot(p2 - p1);
    double d4343 = (p4 - p3).Dot(p4 - p3);
    double d2121 = (p2 - p1).Dot(p2 - p1);

    li->fA = (d1343 * d4321 - d1321 * d4343) / (d2121 * d4343 - d4321 * d4321);
    li->fB = (d1343 + li->fA * d4321) / d4343;

    Vec_t pA = p1 + li->fA * (p2 - p1);
    Vec_t pB = p3 + li->fB * (p4 - p3);

    if (pA == pB) {
      li->fType = LineIntersection::fIntersect;
    } else {
      li->fType = LineIntersection::fNoIntersect;
    }
  }
  return li;
}

bool Polygon::IsPointInside(const Vec_t &p) const
{
  int cn = 0;

  for (size_t i = 0; i < fInd.size(); i++) {
    size_t k = (i + 1) % fInd.size();
    Line l{{(*fVert)[fInd[i]], (*fVert)[fInd[k]]}};
    if (l.IsPointOnLine(p)) {
      return true;
    }
  }

  Line pl = Line{{p, (*fVert)[fInd[0]]}};

  for (size_t i = 0; i < fInd.size(); i++) {
    size_t k = (i + 1) % fInd.size();

    Line l{{(*fVert)[fInd[i]], (*fVert)[fInd[k]]}};

    LineIntersection *li = pl.Intersect(l);

    if (li->fType == LineIntersection::fIntersect) {
      bool upwardEdge = ((*fVert)[fInd[0]] - p).Cross((*fVert)[fInd[k]] - (*fVert)[fInd[i]]).Dot(fNorm) > 0;

      if (upwardEdge) {

        if (li->fA > kTolerance && li->fB >= -kTolerance && li->fB < 1 - kTolerance) {
          cn++;
        }
      } else {
        if (li->fA > kTolerance && li->fB > kTolerance && li->fB <= 1 + kTolerance) {
          cn++;
        }
      }
    }
  }

  if (cn & 1) return true;
  return false;
}

void Polygon::Extent(Precision x[2], Precision y[2], Precision z[2])
{
  x[0] = x[1] = (*fVert)[fInd[0]].x();
  y[0] = y[1] = (*fVert)[fInd[0]].y();
  z[0] = z[1] = (*fVert)[fInd[0]].z();
  for (auto i : fInd) {
    if ((*fVert)[i].x() > x[1]) x[1] = (*fVert)[i].x();
    if ((*fVert)[i].x() < x[0]) x[0] = (*fVert)[i].x();

    if ((*fVert)[i].y() > y[1]) y[1] = (*fVert)[i].y();
    if ((*fVert)[i].x() < y[0]) y[0] = (*fVert)[i].y();

    if ((*fVert)[i].z() > z[1]) z[1] = (*fVert)[i].z();
    if ((*fVert)[i].z() < z[0]) z[0] = (*fVert)[i].z();
  }
}

#ifndef VECCORE_CUDA
struct PolygonIntersection *Polygon::Intersect(const Polygon &clipper)
{
  PolygonIntersection *pi = new PolygonIntersection();

  if (fNorm != clipper.fNorm && fNorm != -clipper.fNorm) {
    // subject and clipper polygons have different normal

    Vec_t startPoint = Vec_t(std::numeric_limits<double>::infinity());
    Vec_t endPoint   = Vec_t(-std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < clipper.fInd.size(); i++) {
      if (fNorm.Dot(clipper.fSides[i]) == 0.) {
        // clipper line parallel to the plane

      } else {
        // single point of intersection possible

        // store just the start and end point (2 boundaries) of all intersections on the plane
        double d = ((*fVert)[fInd[0]] - (*clipper.fVert)[clipper.fInd[i]]).Dot(fNorm) / (fNorm.Dot(clipper.fSides[i]));
        Vec_t intersection_pt = d * clipper.fSides[i] + (*clipper.fVert)[clipper.fInd[i]];
        // std::cerr << intersection_pt << '\n';
        if (intersection_pt.x() < startPoint.x() ||
            (intersection_pt.x() == startPoint.x() && intersection_pt.y() < startPoint.y()) ||
            ((intersection_pt.x() == startPoint.x() && intersection_pt.y() == startPoint.y() &&
              intersection_pt.y() < startPoint.y()))) {
          startPoint = intersection_pt;
        }

        if (intersection_pt.x() > endPoint.x() ||
            (intersection_pt.x() == endPoint.x() && intersection_pt.y() > endPoint.y()) ||
            ((intersection_pt.x() == endPoint.x() && intersection_pt.y() == endPoint.y() &&
              intersection_pt.y() > endPoint.y()))) {
          endPoint = intersection_pt;
        }
      }
    }

    // std::cerr << startPoint << ' ' << endPoint;
    vector_t<double> as;

    Line l1{{startPoint, endPoint}}; // clipper line

    bool noIntersection = true;

    for (size_t j = 0; j < fInd.size(); j++) {
      // find the intersection with each of subject's lines
      Line l2{{(*fVert)[fInd[j]], (*fVert)[fInd[(j + 1) % fInd.size()]]}}; // subject line

      LineIntersection *li = l1.Intersect(l2);

      if (li->fType == LineIntersection::fIntersect) {
        if (-kTolerance <= li->fA && li->fA <= 1 + kTolerance && -kTolerance <= li->fB && li->fB <= 1 + kTolerance) {
          as.push_back(li->fA);
          noIntersection = false;
        }
      }
    }

    if (noIntersection) {
      if (IsPointInside(startPoint)) {
        pi->fLines.push_back(Line{{startPoint, endPoint}});
      }
      return pi;
    }

    std::sort(as.begin(), as.end());
    as.erase(std::unique(as.begin(), as.end()), as.end());

    bool lastLine = false;
    for (size_t i = 0; i < as.size() - 1; i++) {
      double middle  = (as[i] + as[i + 1]) / 2;
      Vec_t midpoint = l1.fPts[0] + middle * (l1.fPts[1] - l1.fPts[0]);
      if (IsPointInside(midpoint)) {
        pi->fLines.push_back(
            Line{{l1.fPts[0] + as[i] * (l1.fPts[1] - l1.fPts[0]), l1.fPts[0] + as[i + 1] * (l1.fPts[1] - l1.fPts[0])}});
        lastLine = true;
      } else {
        if (!lastLine) {
          pi->fPoints.push_back(l1.fPts[0] + as[i] * (l1.fPts[1] - l1.fPts[0]));
        }
        lastLine = false;
      }
    }

    if (!lastLine) {
      pi->fPoints.push_back(l1.fPts[0] + as[as.size() - 1] * (l1.fPts[1] - l1.fPts[0]));
    }

  } else {
    // results in polygon set

    struct GreinerHormannVertex {
      Vec_t coord;
      GreinerHormannVertex *next     = nullptr;
      GreinerHormannVertex *prev     = nullptr;
      bool intersect                 = false;
      bool entry                     = false;
      bool visited                   = false;
      GreinerHormannVertex *neighbor = nullptr;
      double alpha                   = 0.;
      Vec_t perturbated;

      GreinerHormannVertex() {};
      GreinerHormannVertex(double alpha, Vec_t coord, bool intersect)
          : coord(coord), intersect(intersect), alpha(alpha) {};

      void MakeNeighbor(GreinerHormannVertex *i2)
      {
        this->neighbor = i2;
        i2->neighbor   = this;
      }
    };

    struct GreinerHormannPolygon {
      GreinerHormannVertex *head = nullptr;
      int size                   = 0;

      GreinerHormannPolygon() {};
      GreinerHormannPolygon(const Polygon &poly)
      {
        head       = new GreinerHormannVertex(); // dummy head
        head->prev = head;
        head->next = head;

        GreinerHormannVertex *current;
        for (auto i : poly.fInd) {
          current        = new GreinerHormannVertex();
          current->coord = (*poly.fVert)[i];
          current->next  = head;
          current->prev  = head->prev;

          head->prev->next = current;
          head->prev       = current;
          size++;
        }

        head->prev->next = head->next;
        head->next->prev = head->prev;
        // delete old head
        current = head;
        head    = head->next;
        delete current;
      }

      ~GreinerHormannPolygon()
      {
        GreinerHormannVertex *current = head;
        for (int i = 0; i < size; i++) {
          head = head->next;
          delete current;
          current = head;
        }
      }

      void Insert(GreinerHormannVertex *ins, GreinerHormannVertex *first)
      {
        GreinerHormannVertex *aux = first;
        GreinerHormannVertex *lst = first;
        do {
          lst = lst->next;
        } while (lst->intersect);

        while (aux != lst && aux->alpha < ins->alpha)
          aux = aux->next;

        ins->next       = aux;
        ins->prev       = aux->prev;
        ins->prev->next = ins;
        ins->next->prev = ins;
        size++;
      }

      GreinerHormannVertex *First()
      {
        GreinerHormannVertex *temp = head;
        for (int i = 0; i < size; i++) {
          if (temp->intersect && !temp->visited) {
            return temp;
          }
          temp = temp->next;
        }
        return nullptr;
      }

      void Perturbate(GreinerHormannPolygon *other, const Vec_t &fNorm)
      {

        GreinerHormannVertex *current = this->head;
        do {
          // Line l1{current->coord, current->next->coord};
          GreinerHormannVertex *other_current = other->head;
          do {
            Line l2{{other_current->coord, other_current->next->coord}};
            bool res = l2.IsPointOnLine(current->coord);
            if (res) {
              Vec_t dir = fNorm.Cross(l2.fPts[1] - l2.fPts[0]).Normalized();
              current->coord += dir * Vec_t(1e-7);
              current->perturbated += dir * Vec_t(1e-7);
            }

          } while ((other_current = other_current->next) != other->head);
        } while ((current = current->next) != this->head);

        current = other->head;
        do {
          // Line l1{current->coord, current->next->coord};
          GreinerHormannVertex *other_current = this->head;
          do {
            Line l2{{other_current->coord, other_current->next->coord}};
            bool res = l2.IsPointOnLine(current->coord);
            if (res) {
              Vec_t dir = fNorm.Cross(l2.fPts[1] - l2.fPts[0]).Normalized();
              current->coord += dir * Vec_t(1e-7);
              current->perturbated += dir * Vec_t(1e-7);
            }

          } while ((other_current = other_current->next) != this->head);
        } while ((current = current->next) != other->head);
      }
    };

    GreinerHormannPolygon *clp = new GreinerHormannPolygon(clipper);
    GreinerHormannPolygon *sbj = new GreinerHormannPolygon(*this);

    sbj->Perturbate(clp, fNorm);

    // phase 1
    bool noIntersection               = true;
    GreinerHormannVertex *clp_current = clp->head;
    GreinerHormannVertex *clp_nxt     = clp_current;
    do {
      while ((clp_nxt = clp_nxt->next)->intersect)
        ;
      Line l1{{clp_current->coord, clp_nxt->coord}};
      GreinerHormannVertex *sbj_current = sbj->head;
      GreinerHormannVertex *sbj_nxt     = sbj_current;
      do {
        while ((sbj_nxt = sbj_nxt->next)->intersect)
          ;
        Line l2{{sbj_current->coord, sbj_nxt->coord}};
        LineIntersection *li = l1.Intersect(l2);

        if (li->fType == LineIntersection::fIntersect) {
          if ((kTolerance < li->fA && li->fA < 1 - kTolerance && kTolerance < li->fB && li->fB < 1 - kTolerance)) {
            // insert
            GreinerHormannVertex *i1 =
                new GreinerHormannVertex(li->fA, l1.fPts[0] + li->fA * (l1.fPts[1] - l1.fPts[0]), true);
            GreinerHormannVertex *i2 =
                new GreinerHormannVertex(li->fB, l2.fPts[0] + li->fB * (l2.fPts[1] - l2.fPts[0]), true);
            i1->MakeNeighbor(i2);
            clp->Insert(i1, clp_current);
            sbj->Insert(i2, sbj_current);
            noIntersection = false;
          }
        }

      } while ((sbj_current = sbj_nxt) != sbj->head);
    } while ((clp_current = clp_nxt) != clp->head);

    if (noIntersection) {

      // test clp point
      if (((*this->fVert)[this->fInd[0]] - clp->head->coord).Dot(this->fNorm) == 0 &&
          this->IsPointInside(clp->head->coord)) {
        pi->fPolygons.push_back(clipper);
        return pi;
      }

      // test sbj point
      if (((*clipper.fVert)[clipper.fInd[0]] - sbj->head->coord).Dot(clipper.fNorm) == 0 &&
          clipper.IsPointInside(sbj->head->coord)) {
        pi->fPolygons.push_back(*this);
        return pi;
      }
      return pi;
    }

    // phase 2

    bool status = !this->IsPointInside(clp->head->coord);

    GreinerHormannVertex *temp = clp->head;
    for (int i = 0; i < clp->size; i++) {
      if (temp->intersect) {
        temp->entry = status;
        status      = !status;
      }
      temp = temp->next;
    }

    status = !clipper.IsPointInside(sbj->head->coord);
    temp   = sbj->head;
    for (int i = 0; i < sbj->size; i++) {
      if (temp->intersect) {
        temp->entry = status;
        status      = !status;
      }
      temp = temp->next;
    }

    // phase 3
    GreinerHormannVertex *current;
    while ((current = sbj->First())) {

      vector_t<size_t> ind;
      for (; !current->visited; current = current->neighbor) {

        current->visited = true;
        for (bool forward = current->entry;;) {

          ind.push_back(pi->fVertices.size());
          pi->fVertices.push_back(current->coord - current->perturbated);

          current = forward ? current->next : current->prev;
          if (current->intersect) {
            current->visited = true;
            break;
          }
        }
      }
      pi->fPolygons.push_back({ind.size(), pi->fVertices, ind, false});
    }

    auto it = pi->fPolygons.begin();
    while (it != pi->fPolygons.end()) {
      if (!it->fValid) {
        if (it->fN == 2) {
          // convert to line
          pi->fLines.push_back({{(*it->fVert)[it->fInd[0]], (*it->fVert)[it->fInd[1]]}});
        } else if (it->fN == 1) {
          // convert to point
          pi->fPoints.push_back((*it->fVert)[it->fInd[0]]);
        }

        it = pi->fPolygons.erase(it);
      } else {
        it++;
      }
    }
  }

  return pi;
}
#endif

void Polygon::Init()
{

  // Compute sides
  for (size_t i = 0; i < fN - 1; ++i) {
    fSides[i] = GetVertex(i + 1) - GetVertex(i);
    VECGEOM_ASSERT(fSides[i].Mag2() > kTolerance);
  }
  fSides[fN - 1] = GetVertex(0) - GetVertex(fN - 1);
  VECGEOM_ASSERT(fSides[fN - 1].Mag2() > kTolerance);
  // Compute normal if not already set
  if (!fHasNorm) {
    fNorm = fSides[0].Cross(fSides[1]);
    fNorm.Normalize();
  }
  VECGEOM_ASSERT((fSides[0].Cross(fSides[1])).Dot(fNorm) > 0);
  // Compute convexity if not supplied
  if (!fConvex) {
    fConvex = true;
    for (size_t i = 0; i < fN; ++i) {
      for (size_t k = 0; k < fN - 2; ++k) {
        size_t j = (i + k + 2) % fN; // remaining vertices
        if (fSides[i].Cross(GetVertex(j) - GetVertex(i)).Dot(fNorm) < 0) {
          fConvex = false;
          break;
        }
      }
    }
  }
  // Compute distance to origin
  fDist = -fNorm.Dot(GetVertex(0));
}

void Polygon::Transform(Transformation3D const &tr)
{
  // The polygon must be already initialized and the vertices transformed
  Vec_t temp;
  tr.InverseTransformDirection(fNorm, temp);
  fNorm = temp;
  // Compute sides
  for (size_t i = 0; i < fN - 1; ++i)
    fSides[i] = GetVertex(i + 1) - GetVertex(i);
  fSides[fN - 1] = GetVertex(0) - GetVertex(fN - 1);
  // Compute distance to origin
  fDist = -fNorm.Dot(GetVertex(0));
}

void Polyhedron::Transform(Transformation3D const &tr)
{
  // Transform vertices
  for (size_t i = 0; i < fVert.size(); ++i) {
    Vec_t temp;
    tr.InverseTransform(fVert[i], temp);
    fVert[i] = temp;
  }
  for (size_t i = 0; i < fPolys.size(); ++i)
    fPolys[i].Transform(tr);
}

void Polyhedron::AddPolygon(Polygon &poly, bool triangulate)
{
  poly.Init();

  if (triangulate) {
    size_t before = fPolys.size();
    poly.TriangulatePolygon(fPolys);
    for (size_t i = before; i < fPolys.size(); i++) {
      fPolys[i].Init();
    }
  } else {
    fPolys.push_back(poly);
  }
}

void FillBoxPolyhedron(Vec_t const &box, Polyhedron &polyh)
{
  polyh.Reset(8, 6);
  vector_t<Vec_t> &vert    = polyh.fVert;
  vector_t<Polygon> &polys = polyh.fPolys;

  vert          = {{-box[0], -box[1], -box[2]}, {-box[0], box[1], -box[2]}, {box[0], box[1], -box[2]},
                   {box[0], -box[1], -box[2]},  {-box[0], -box[1], box[2]}, {-box[0], box[1], box[2]},
                   {box[0], box[1], box[2]},    {box[0], -box[1], box[2]}};
  polys         = {{4, vert, {0., 0., -1.}}, {4, vert, {0., 0., 1.}}, {4, vert, {-1., 0., 0.}},
                   {4, vert, {0., 1., 0.}},  {4, vert, {1., 0., 0.}}, {4, vert, {0., -1., 0.}}};
  polys[0].fInd = {0, 1, 2, 3};
  polys[1].fInd = {4, 7, 6, 5};
  polys[2].fInd = {0, 4, 5, 1};
  polys[3].fInd = {1, 5, 6, 2};
  polys[4].fInd = {2, 6, 7, 3};
  polys[5].fInd = {3, 7, 4, 0};
  for (size_t i = 0; i < 6; ++i)
    polys[i].Init();
}

EPlaneXing_t PlaneXing(Plane const &pl1, Plane const &pl2, Vector3D<Precision> &point, Vector3D<Precision> &direction)
{
  direction        = pl1.fNorm.Cross(pl2.fNorm);
  const double det = direction.Mag2();
  if (Abs(det) < kTolerance) {
    // The 2 planes are parallel, let's find the distance between them
    const double d12 = Abs(Abs(pl1.fDist) - Abs(pl2.fDist));
    if (d12 < kTolerance) {
      if (pl1.fDist * pl2.fDist * pl1.fNorm.Dot(pl2.fNorm) > 0.) return kIdentical;
    }
    return kParallel;
  }
  // The planes do intersect
  point = ((pl1.fDist * direction.Cross(pl2.fNorm) - pl2.fDist * direction.Cross(pl1.fNorm))) / det;
  direction.Normalize();
  return kIntersecting;
}

EBodyXing_t PolygonXing(Polygon const &poly1, Polygon const &poly2, Line *line)
{
  using Vec_t = Vector3D<Precision>;
  using vecCore::math::CopySign;
  using vecCore::math::Max;
  using vecCore::math::Min;
  using vecCore::math::Sqrt;

  Vec_t point, direction;
  EPlaneXing_t crossing = PlaneXing(Plane(poly1.fNorm, poly1.fDist), Plane(poly2.fNorm, poly2.fDist), point, direction);
  if (crossing == kParallel) return kDisjoint;

  if (crossing == kIdentical) {
    // We use the separate axis theorem
    // loop segments of 1
    for (size_t i = 0; i < poly1.fN; ++i) {
      // loop vertices of 2
      bool outside = false;
      for (size_t j = 0; j < poly2.fN; ++j) {
        outside = poly1.fNorm.Dot((poly2.GetVertex(j) - poly1.GetVertex(i)).Cross(poly1.fSides[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    // loop segments of 2
    for (size_t i = 0; i < poly2.fN; ++i) {
      // loop vertices of 1
      bool outside = false;
      for (size_t j = 0; j < poly1.fN; ++j) {
        outside = poly2.fNorm.Dot((poly1.GetVertex(j) - poly2.GetVertex(i)).Cross(poly2.fSides[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    return kTouching;
  }

  // The polygons do cross each other along a line
  if (!(poly1.fConvex | poly2.fConvex)) return kDisjoint; // cannot solve yet non-convex case
  double smin1 = InfinityLength<double>();
  double smax1 = -InfinityLength<double>();
  for (size_t i = 0; i < poly1.fN; ++i) {
    Vec_t crossdirs      = poly1.fSides[i].Cross(direction);
    double mag2crossdirs = crossdirs.Mag2();
    if (mag2crossdirs < kTolerance) continue; // Crossing line parallel to edge
    Vec_t crosspts = (point - poly1.GetVertex(i)).Cross(direction);
    if (crossdirs.Dot(crosspts) < 0) continue;
    if (crosspts.Mag2() < mag2crossdirs + kTolerance) {
      // Crossing line hits the current edge
      crosspts      = (point - poly1.GetVertex(i)).Cross(poly1.fSides[i]);
      double distsq = CopySign<double>(crosspts.Mag2() / mag2crossdirs, crosspts.Dot(crossdirs));
      smin1         = Min(smin1, distsq);
      smax1         = Max(smax1, distsq);
    }
  }
  if (smax1 <= smin1) return kDisjoint;

  double smin2 = InfinityLength<double>();
  double smax2 = -InfinityLength<double>();
  for (size_t i = 0; i < poly2.fN; ++i) {
    Vec_t crossdirs      = poly2.fSides[i].Cross(direction);
    double mag2crossdirs = crossdirs.Mag2();
    if (mag2crossdirs < kTolerance) continue; // Crossing line parallel to edge
    Vec_t crosspts = (point - poly2.GetVertex(i)).Cross(direction);
    if (crossdirs.Dot(crosspts) < 0) continue;
    if (crosspts.Mag2() < mag2crossdirs + kTolerance) {
      // Crossing line hits the current edge
      crosspts      = (point - poly2.GetVertex(i)).Cross(poly2.fSides[i]);
      double distsq = CopySign<double>(crosspts.Mag2() / mag2crossdirs, crosspts.Dot(crossdirs));
      smin2         = Min(smin2, distsq);
      smax2         = Max(smax2, distsq);
    }
  }
  if (smax2 <= smin2) return kDisjoint;
  if (smin2 - smax1 > -kTolerance || smin1 - smax2 > -kTolerance) return kDisjoint;
  if (line != nullptr) {
    double dmin = Max(smin1, smin2);
    double dmax = Min(smax1, smax2);
    VECGEOM_ASSERT(dmax - dmin > -kTolerance);
    line->fPts[0] = point + direction * CopySign<double>(Sqrt(Abs(dmin)), dmin);
    line->fPts[1] = point + direction * CopySign<double>(Sqrt(Abs(dmax)), dmax);
  }
  return kOverlapping;
}

EBodyXing_t PolyhedronXing(Polyhedron const &polyh1, Polyhedron const &polyh2, vector_t<Line> &lines)
{
  // We assume PolyhedronCollision was called and the polyhedra do intersect. The polihedra are already transformed.
  using vecCore::math::Max;
  Line line;
  EBodyXing_t result = kDisjoint;
  for (const auto &poly1 : polyh1.fPolys) {
    for (const auto &poly2 : polyh2.fPolys) {
      EBodyXing_t crossing = PolygonXing(poly1, poly2, &line);
      result               = Max(result, crossing);
      if (crossing == kOverlapping) lines.push_back(line);
    }
  }
  return result;
}

EBodyXing_t BoxCollision(Vector3D<Precision> const &box1, Transformation3D const &tr1, Vector3D<Precision> const &box2,
                         Transformation3D const &tr2)
{
  // A fast check if the bounding spheres touch
  using Vec_t = Vector3D<Precision>;
  using vecCore::math::Max;
  using vecCore::math::Min;

  Vec_t orig1 = tr1.Translation();
  Vec_t orig2 = tr2.Translation();
  double r1sq = box1.Mag2();
  double r2sq = box2.Mag2();
  double dsq  = (orig2 - orig1).Mag2();
  if (dsq > r1sq + r2sq + 2. * Sqrt(r1sq * r2sq)) return kDisjoint;

  if (!tr1.HasRotation() && !tr2.HasRotation()) {
    // Aligned boxes case
    Vector3D<double> eps1 = (orig2 - box2) - (orig1 + box1);
    Vector3D<double> eps2 = (orig1 - box1) - (orig2 + box2);
    double deps           = Max(eps1.Max(), eps2.Max());
    if (deps > kTolerance)
      return kDisjoint;
    else if (deps > -kTolerance)
      return kTouching;
    return kOverlapping;
  }
  // General case: use separating plane theorem (3D version of SAT)

  // A lambda computing min for the i component
  // compute matrix to go from 2 to 1
  Transformation3D tr12;
  tr12 = tr1.Inverse();
  tr12.MultiplyFromRight(tr2); // Relative transformation of 2 in local coordinates of 1
  // Fill mesh of points for 2
  const Vec_t mesh2[8] = {{-box2[0], -box2[1], -box2[2]}, {-box2[0], box2[1], -box2[2]}, {box2[0], box2[1], -box2[2]},
                          {box2[0], -box2[1], -box2[2]},  {-box2[0], -box2[1], box2[2]}, {-box2[0], box2[1], box2[2]},
                          {box2[0], box2[1], box2[2]},    {box2[0], -box2[1], box2[2]}};
  Vec_t mesh[8];
  for (auto i = 0; i < 8; ++i)
    tr12.InverseTransform(mesh2[i], mesh[i]);

  // Check mesh2 against faces of 1
  const double maxx2 = Max(Max((mesh[0].x(), mesh[1].x()), Max(mesh[2].x(), mesh[3].x())),
                           Max((mesh[4].x(), mesh[5].x()), Max(mesh[6].x(), mesh[7].x())));
  if (maxx2 < -box1[0] - kTolerance) return kDisjoint;
  const double minx2 = Min(Min((mesh[0].x(), mesh[1].x()), Min(mesh[2].x(), mesh[3].x())),
                           Min((mesh[4].x(), mesh[5].x()), Min(mesh[6].x(), mesh[7].x())));
  if (minx2 > box1[0] + kTolerance) return kDisjoint;

  const double maxy2 = Max(Max((mesh[0].y(), mesh[1].y()), Max(mesh[2].y(), mesh[3].y())),
                           Max((mesh[4].y(), mesh[5].y()), Max(mesh[6].y(), mesh[7].y())));
  if (maxy2 < -box1[1] - kTolerance) return kDisjoint;
  const double miny2 = Min(Min((mesh[0].y(), mesh[1].y()), Min(mesh[2].y(), mesh[3].y())),
                           Min((mesh[4].y(), mesh[5].y()), Min(mesh[6].y(), mesh[7].y())));
  if (miny2 > box1[1] + kTolerance) return kDisjoint;

  const double maxz2 = Max(Max((mesh[0].z(), mesh[1].z()), Max(mesh[2].z(), mesh[3].z())),
                           Max((mesh[4].z(), mesh[5].z()), Max(mesh[6].z(), mesh[7].z())));
  if (maxz2 < -box1[2] - kTolerance) return kDisjoint;
  const double minz2 = Min(Min((mesh[0].z(), mesh[1].z()), Min(mesh[2].z(), mesh[3].z())),
                           Min((mesh[4].z(), mesh[5].z()), Min(mesh[6].z(), mesh[7].z())));
  if (minz2 > box1[2] + kTolerance) return kDisjoint;

  // Fill mesh of points for 2
  const Vec_t mesh1[8] = {{-box1[0], -box1[1], -box1[2]}, {-box1[0], box1[1], -box1[2]}, {box1[0], box1[1], -box1[2]},
                          {box1[0], -box1[1], -box1[2]},  {-box1[0], -box1[1], box1[2]}, {-box1[0], box1[1], box1[2]},
                          {box1[0], box1[1], box1[2]},    {box1[0], -box1[1], box1[2]}};

  Transformation3D tr21;
  tr21 = tr2.Inverse();
  tr21.MultiplyFromRight(tr1); // Relative transformation of 2 in local coordinates of 1
  for (auto i = 0; i < 8; ++i)
    tr21.InverseTransform(mesh1[i], mesh[i]);

  // Check mesh2 against faces of 1
  const double maxx1 = Max(Max((mesh[0].x(), mesh[1].x()), Max(mesh[2].x(), mesh[3].x())),
                           Max((mesh[4].x(), mesh[5].x()), Max(mesh[6].x(), mesh[7].x())));
  if (maxx1 < -box2[0] - kTolerance) return kDisjoint;
  const double minx1 = Min(Min((mesh[0].x(), mesh[1].x()), Min(mesh[2].x(), mesh[3].x())),
                           Min((mesh[4].x(), mesh[5].x()), Min(mesh[6].x(), mesh[7].x())));
  if (minx1 > box2[0] + kTolerance) return kDisjoint;

  const double maxy1 = Max(Max((mesh[0].y(), mesh[1].y()), Max(mesh[2].y(), mesh[3].y())),
                           Max((mesh[4].y(), mesh[5].y()), Max(mesh[6].y(), mesh[7].y())));
  if (maxy1 < -box2[1] - kTolerance) return kDisjoint;
  const double miny1 = Min(Min((mesh[0].y(), mesh[1].y()), Min(mesh[2].y(), mesh[3].y())),
                           Min((mesh[4].y(), mesh[5].y()), Min(mesh[6].y(), mesh[7].y())));
  if (miny1 > box2[1] + kTolerance) return kDisjoint;

  const double maxz1 = Max(Max((mesh[0].z(), mesh[1].z()), Max(mesh[2].z(), mesh[3].z())),
                           Max((mesh[4].z(), mesh[5].z()), Max(mesh[6].z(), mesh[7].z())));
  if (maxz1 < -box2[2] - kTolerance) return kDisjoint;
  const double minz1 = Min(Min((mesh[0].z(), mesh[1].z()), Min(mesh[2].z(), mesh[3].z())),
                           Min((mesh[4].z(), mesh[5].z()), Min(mesh[6].z(), mesh[7].z())));
  if (minz1 > box2[2] + kTolerance) return kDisjoint;

  return kOverlapping;
}

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
