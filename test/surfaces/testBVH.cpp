#include "test/benchmark/ArgParser.h"
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/surfaces/Model.h>
#include <VecGeom/surfaces/BrepHelper.h>
#include <VecGeom/surfaces/Navigator.h>
#include <VecGeom/surfaces/BVHSurfNavigator.h>
#include <VecGeom/surfaces/bvh/BVHsurfCreator.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/volumes/utilities/VolumeUtilities.h>
#include <VecGeom/navigation/BVHNavigator.h>

#ifdef VECGEOM_GDML
#include <persistency/gdml/source/include/Frontend.h>
#endif

using namespace vecgeom;
using Real_t     = double;
using BrepHelper = vgbrep::BrepHelper<Real_t>;
using SurfData   = vgbrep::SurfData<Real_t>;
using vecCore::math::Abs;
// generation of rays etc is based on vecgeom::Precision as these are used also in the solid model
using Vec3D  = vecgeom::Vector3D<vecgeom::Precision>;
using Vec3Dc = Precision[3];

//==================================================================================
int LoadGDML(const char *gdml_name, bool ongpu, int min_per_scene, double mmunit = 1)
{
#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#else
  GeoManager::Instance().SetMinPerScene(min_per_scene);
  auto load = vgdml::Frontend::Load(gdml_name, false, mmunit);
  if (!load) return 2;
#endif

  auto world = GeoManager::Instance().GetWorld();
  if (!world) return 3;
  vecgeom::cxx::BVHManager::Init();
  return 0;
}

//==================================================================================
int main(int argc, char *argv[])
{
  OPTION_STRING(gdml_name, "default.gdml");
  OPTION_INT(nrays, 10000);
  OPTION_INT(debug, 0);
  OPTION_INT(verbosity, 0);
  OPTION_INT(min_per_scene, 1000);
  OPTION_INT(ongpu, 1);
  OPTION_STRING(bvh_stats_file, "");
  OPTION_STRING(bvh_dump_dir, "");
  OPTION_DOUBLE(mmunit, 1);
  std::vector<double> default_point = {vecgeom::InfinityLength<Precision>(), vecgeom::InfinityLength<Precision>(),
                                       vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(double, point, default_point);
  std::vector<double> default_direction = {0., 0., 0.};
  OPTION_VECTOR(double, direction, default_direction);
  OPTION_VECTOR(double, max_world, default_point);
  std::vector<double> default_min_world = {-vecgeom::InfinityLength<Precision>(), -vecgeom::InfinityLength<Precision>(),
                                           -vecgeom::InfinityLength<Precision>()};
  OPTION_VECTOR(double, min_world, default_min_world);
  VECGEOM_ASSERT(point.size() == 3 && direction.size() == 3);
  VECGEOM_ASSERT(min_world.size() == 3 && default_min_world.size() == 3);

  // transform to Vec3D for further handling
  Vec3D point_3D     = {point[0], point[1], point[2]};
  Vec3D direction_3D = {direction[0], direction[1], direction[2]};
  if (direction_3D.Mag2() > 0) direction_3D.Normalize();

  bool use_provided_point = (direction_3D.Mag2() != 0) || (point_3D.Mag2() < vecgeom::InfinityLength<Precision>());
  if (use_provided_point) {
    // check if direction is normalized
    VECGEOM_ASSERT(direction_3D.IsNormalized());
    nrays = 1;
    if (debug)
      std::cout << "Tracking single ray with point " << point_3D << " and direction " << direction_3D << std::endl;
  }

  vecgeom::logger().level(vecgeom::LogLevel::info);
  Stopwatch timer;
  // Load the geometry
  timer.Start();
  bool load = LoadGDML(gdml_name.c_str(), ongpu, min_per_scene, mmunit);
  if (load > 0) return load;
  auto time_load = timer.Stop();
  std::cout << "Geometry loading: " << time_load << " [s]\n";

  BrepHelper::Instance().SetVerbosity(verbosity);

  timer.Start();
  // Conversion to the surface model
  if (!BrepHelper::Instance().Convert()) return 1;
  auto time_surf_dist = timer.Stop();
  BrepHelper::Instance().PrintSurfData();
  std::cout << "Conversion time to surface model: " << time_surf_dist << " [s]\n";

  auto const &surfdata = BrepHelper::Instance().GetSurfData();

  // BVH Benchmark ==================================================================
  // Per LV, generate points and directions, call and time the BVH
  auto num_lvol = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
  Vec3D *points, *dirs;
  points                 = new Vec3D[nrays];
  dirs                   = new Vec3D[nrays];
  double *solids_times   = new double[num_lvol];
  double *surfaces_times = new double[num_lvol];

  long *solids_total_children_visits   = new long[num_lvol]();
  long *surfaces_total_children_visits = new long[num_lvol]();
  long *solids_total_leaf_visits       = new long[num_lvol]();
  long *surfaces_total_leaf_visits     = new long[num_lvol]();
  long *solids_total_cut_nodes         = new long[num_lvol]();
  long *surfaces_total_cut_nodes       = new long[num_lvol]();
  long *solids_total_stacked_nodes     = new long[num_lvol]();
  long *surfaces_total_stacked_nodes   = new long[num_lvol]();

  for (uint lvol_id = 0; lvol_id < num_lvol; lvol_id++) {

    const auto *bvh_sol = vecgeom::BVHManager::GetBVH(lvol_id);
    const auto *bvh_sur = &surfdata.fBVH[surfdata.fShells[lvol_id].fBVH];

    if (bvh_sol == nullptr) continue;

    // Generate random points and directions inside the volume
    Vec3D amin, amax;
    auto lvol = GeoManager::Instance().GetLogicalVolume(lvol_id);
    lvol->GetUnplacedVolume()->Extent(amin, amax);

    volumeUtilities::FillRandomPoints(amin, amax, points, nrays);
    volumeUtilities::FillRandomDirections(dirs, nrays);

    for (int i = 0; i < nrays; i++) {
      // Solids:
      auto max_step           = vecgeom::kInfLength;
      long hitcandidate_index = -1;
      long last_exited_id     = -1;
      long visited_children   = 0;
      long visited_leaves     = 0;
      long cut_nodes          = 0;
      long stacked_nodes      = 0;
      timer.Start();
      bvh_sol->CheckDaughterIntersectionsBenchmark<BVHNavigator>(points[i], dirs[i], max_step, last_exited_id,
                                                                 hitcandidate_index, visited_children, visited_leaves,
                                                                 cut_nodes, stacked_nodes);
      auto bvh_time_sol = timer.Stop();
      solids_total_children_visits[lvol_id] += visited_children;
      solids_total_leaf_visits[lvol_id] += visited_leaves;
      solids_total_cut_nodes[lvol_id] += cut_nodes;
      solids_total_stacked_nodes[lvol_id] += stacked_nodes;
      // Surfaces:
      max_step         = vecgeom::kInfLength;
      visited_children = 0;
      visited_leaves   = 0;
      cut_nodes        = 0;
      stacked_nodes    = 0;
      bvh_sur->template CheckDaughterIntersectionsBenchmark<vgbrep::protonav::BVHSurfNavigator<vecgeom::Precision>,
                                                            vecgeom::Precision>(
          points[i], dirs[i], max_step, last_exited_id, hitcandidate_index, visited_children, visited_leaves, cut_nodes,
          stacked_nodes);
      auto bvh_time_sur = timer.Stop();
      surfaces_total_children_visits[lvol_id] += visited_children;
      surfaces_total_leaf_visits[lvol_id] += visited_leaves;
      surfaces_total_cut_nodes[lvol_id] += cut_nodes;
      surfaces_total_stacked_nodes[lvol_id] += stacked_nodes;

      solids_times[lvol_id]   = bvh_time_sol;
      surfaces_times[lvol_id] = bvh_time_sur;
    }
  }

  // double total_cut{0}, total_stacked{0};
  // for(uint i=0; i<num_lvol; i++)
  // {
  //   total_cut+=solids_total_cut_nodes[i];
  //   total_stacked+=solids_total_stacked_nodes[i];
  // }
  // printf("Solids cut nodes avg proportion: %lf\n", total_cut/total_stacked);

  // total_cut=0;total_stacked=0;
  // for(uint i=0; i<num_lvol; i++)
  // {
  //   total_cut+=surfaces_total_cut_nodes[i];
  //   total_stacked+=surfaces_total_stacked_nodes[i];
  // }
  // printf("Surfaces cut nodes avg proportion: %lf\n", total_cut/total_stacked);

  // Stats Output ==================================================================

  if (bvh_stats_file != "") {
    // Output file
    std::fstream bvh_file(bvh_stats_file, std::ios::out);

    auto num_lvol   = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();
    int aRootNChild = 0;
    int aDepth      = 0;
    const int *aNChild{nullptr};
    double *times;
    long *total_children_visits;
    long *total_leaf_visits;
    long *total_stacked_nodes;
    long *total_cut_nodes;

    // Writing the stats in JSON format
    // Two arrays containing the BVH stats for each volume
    bvh_file << "{" << std::endl;

    // Store some general information about the geometry
    bvh_file << "\"num_lvol\": " << num_lvol << "," << std::endl;

    for (int i = 0; i <= 1; i++) {
      // Print data for solids first, then surfaces
      if (i == 0)
        bvh_file << "\"solids\": [" << std::endl;
      else
        bvh_file << "\"surfaces\": [" << std::endl;

      bool first{true};
      for (uint lvol_id = 0; lvol_id < num_lvol; lvol_id++) {
        if (i == 0) // Solids
        {
          auto bvh = vecgeom::BVHManager::GetBVH(lvol_id);
          if (bvh == nullptr) continue;
          aRootNChild           = bvh->GetRootNChild();
          aDepth                = bvh->GetDepth();
          aNChild               = bvh->GetNChild();
          times                 = solids_times;
          total_children_visits = solids_total_children_visits;
          total_leaf_visits     = solids_total_leaf_visits;
          total_stacked_nodes   = solids_total_stacked_nodes;
          total_cut_nodes       = solids_total_cut_nodes;
        } else // Surfaces
        {
          auto bvh = &(surfdata.fBVH[surfdata.fShells[lvol_id].fBVH]);
          if (bvh == nullptr) continue;
          aRootNChild           = bvh->GetRootNChild();
          aDepth                = bvh->GetDepth();
          aNChild               = bvh->GetNChild();
          times                 = surfaces_times;
          total_children_visits = surfaces_total_children_visits;
          total_leaf_visits     = surfaces_total_leaf_visits;
          total_stacked_nodes   = surfaces_total_stacked_nodes;
          total_cut_nodes       = surfaces_total_cut_nodes;
        }

        if (aRootNChild != 0) {
          // For the surfaces, we can decide wether to include the BVH only for LVs with children
          if (i == 1) {
            // Get the shell for this volume
            auto shell = surfdata.fShells[lvol_id];
            if (shell.fNEnteringSurfaces == 0) {
              continue;
            }
          }

          if (!first) bvh_file << "," << std::endl;
          // Start of JSON dict
          bvh_file << "{" << std::endl;

          bvh_file << "\"vol_id\": " << lvol_id << "," << std::endl;
          bvh_file << "\"num_children\": " << aRootNChild << "," << std::endl;
          auto depth = aDepth;
          bvh_file << "\"depth\": " << depth << "," << std::endl;
          // Compute size of nodes array
          auto num_nodes = (2 << depth) - 1;
          // Write the nodes array to the file
          const int *nodes = aNChild;
          bvh_file << "\"nodes\": [";
          for (int n = 0; n < num_nodes; n++)
            if (n != num_nodes - 1)
              bvh_file << nodes[n] << ", ";
            else
              bvh_file << nodes[n] << "]"
                       << "," << std::endl;
          bvh_file << "\"time\": " << times[lvol_id] << "," << std::endl;
          bvh_file << "\"total_children_visits\": " << total_children_visits[lvol_id] << "," << std::endl;
          bvh_file << "\"avg_children_visits\": " << (double)total_children_visits[lvol_id] / nrays << "," << std::endl;
          bvh_file << "\"avg_leaf_visits\": " << (double)total_leaf_visits[lvol_id] / nrays << "," << std::endl;
          bvh_file << "\"total_stacked_nodes\": " << (double)total_stacked_nodes[lvol_id] << "," << std::endl;
          bvh_file << "\"total_cut_nodes\": " << (double)total_cut_nodes[lvol_id] << std::endl;

          // End of JSON dict
          bvh_file << "}";
          first = false;
        }
      }

      // End of JSON array
      if (i == 0)
        bvh_file << "]," << std::endl;
      else
        bvh_file << "]" << std::endl;
    }

    // End of JSON
    bvh_file << "}" << std::endl;

    bvh_file.close();

    printf("Saved global BVH stats to: %s\n", bvh_stats_file.c_str());
  }

  if (bvh_dump_dir != "") {
    for (uint lvol_id = 0; lvol_id < num_lvol; lvol_id++) {
      auto lvol       = GeoManager::Instance().GetLogicalVolume(lvol_id);
      const auto &bvh = surfdata.fBVH[surfdata.fShells[lvol_id].fBVH];
      // if(bvh == nullptr)
      //   continue;
      if (bvh.GetNChild() == 0) continue;
      std::stringstream filename;
      filename << bvh_dump_dir << "/" << lvol->GetLabel() << ".bin";
      vgbrep::bvh::DumpBVH<SurfData::Real_b>(bvh, filename.str().c_str());
    }
    printf("Saved BVH dumps to: %s\n", bvh_dump_dir.c_str());
  }

  // Clear surface data
  BrepHelper::Instance().ClearData();
  delete[] points;
  delete[] dirs;
  delete[] solids_times;
  delete[] surfaces_times;
  delete[] solids_total_children_visits;
  delete[] surfaces_total_children_visits;
  delete[] solids_total_leaf_visits;
  delete[] surfaces_total_leaf_visits;
}