#
# GEANT4 SBT Script to test G4Polyhedra
# DCW 25/3/99 First try
# DCW 23/5/99 Add voxel tests
#
# Increment the numbers below to **really** consume CPU time
#

/performance/maxPoints 10000
/performance/repeat 10

#/solid/G4Polyhedra 0 360 6 4 (1.0,1.2,1.4, 1.2) (-1.0,0,1.0,2.0)
/solid/G4Polyhedra 0 360 6 4 (0,1.2,1.4, 0) (-1.0,0,1.0,2.0)

/performance/errorFileName log/polyhedra-4-test/polyhedrap.a1.log
/control/execute usolids/test.sbt

exit

/solid/G4Polyhedra 0 360 6 6 (0,1.2,1.4, 1.2,1.4, 0) (-1.0,0,1.0,2.0, 3, 4)

/performance/errorFileName log/polyhedra-6-test/polyhedrap.a1.log
/control/execute usolids/test.sbt

/solid/G4Polyhedra 0 360 6 9 (0,1.2,1.4, 1.2,1.4, 1.2,1.3, 1.4 0) (-1.0,0,1.0,2.0, 3, 4, 5, 6, 7)

/performance/errorFileName log/polyhedra-9-test/polyhedrap.a1.log
/control/execute usolids/test.sbt

exit


# ## Polyhedra 12 #######################################################################
#
/solid/G4Polyhedra 0 360 6 12 (0,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,40)
/performance/errorFileName log/polyhedra-12-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 22 #######################################################################
#
/solid/G4Polyhedra 0 360 6 22 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,80)
/performance/errorFileName log/polyhedra-22-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 32 #######################################################################
#
/solid/G4Polyhedra 0 360 6 32 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,120)
/performance/errorFileName log/polyhedra-32-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 42 #######################################################################
#
/solid/G4Polyhedra 0 360 6 42 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,122,127,132,140,145,146,147,150,152,155,160)
/performance/errorFileName log/polyhedra-42-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 52 #######################################################################
#
/solid/G4Polyhedra 0 360 6 52 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,122,127,132,140,145,146,147,150,152,155,162,167,172,180,185,186,187,190,192,195,200)
/performance/errorFileName log/polyhedra-52-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 62 #######################################################################
#
/solid/G4Polyhedra 0 360 6 62 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,122,127,132,140,145,146,147,150,152,155,162,167,172,180,185,186,187,190,192,195,202,207,212,220,225,226,227,230,232,235,240)
/performance/errorFileName log/polyhedra-62-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 72 #######################################################################
#
/solid/G4Polyhedra 0 360 6 72 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,122,127,132,140,145,146,147,150,152,155,162,167,172,180,185,186,187,190,192,195,202,207,212,220,225,226,227,230,232,235,242,247,252,260,265,266,267,270,272,275,280)
/performance/errorFileName log/polyhedra-72-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 82 #######################################################################
#
/solid/G4Polyhedra 0 360 6 82 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,122,127,132,140,145,146,147,150,152,155,162,167,172,180,185,186,187,190,192,195,202,207,212,220,225,226,227,230,232,235,242,247,252,260,265,266,267,270,272,275,282,287,292,300,305,306,307,310,312,315,320)
/performance/errorFileName log/polyhedra-82-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#
# ## Polyhedra 92 #######################################################################
#
/solid/G4Polyhedra 0 360 6 92 (0,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,2,5,5,15,8,8.1,8.1,8,10,10,0) (1,2,7,12,20,25,26,27,30,32,35,42,47,52,60,65,66,67,70,72,75,82,87,92,100,105,106,107,110,112,115,122,127,132,140,145,146,147,150,152,155,162,167,172,180,185,186,187,190,192,195,202,207,212,220,225,226,227,230,232,235,242,247,252,260,265,266,267,270,272,275,282,287,292,300,305,306,307,310,312,315,322,327,332,340,345,346,347,350,352,355,360)
/performance/errorFileName log/polyhedra-92-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
#

exit




/test/maxPoints 10000
/voxel/maxVoxels 500
#
/test/maxPoints 1000
/voxel/maxVoxels 50
#
# --- polyhedra.a1.log
# Start with a nice and easy case
#


/solid/G4Polyhedra 0 360 6 8 4 (1.0,1.2,1.4,1.2) (-1.0,-1.0,1.0,1.0)
/test/errorFileName  log/polyhedra.a1.log
/test/run
/voxel/errorFileName log/polyhedrav.a1.log
/voxel/run
#
# --- polyhedra.a2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/polyhedra.a2.log
/test/run
#
#
# --- polyhedra.b1.log
# Now add a phi slice
#
/solid/G4Polyhedra 0 90 8 4 (1.0,1.2,1.4,1.2) (-1.0,-1.0,1.0,1.0)
/test/gridSizes 0 0 0 m
/test/errorFileName  log/polyhedra.b1.log
/test/run
/voxel/errorFileName log/polyhedrav.b1.log
/voxel/run
#
# --- polyhedra.b2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/polyhedra.b2.log
/test/run
#
# --- polyhedra.c1.log
# Build a much more complicated polyhedra
#
/solid/G4Polyhedra 0 360 6 6 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 
/test/gridSizes 0 0 0 m
/test/errorFileName  log/polyhedra.c1.log
/test/run
/voxel/errorFileName log/polyhedrav.c1.log
/voxel/run
#
# --- polyhedra.b2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/polyhedra.c2.log
/test/run
#
# --- polyhedra.d1.log
# Build a much more complicated polyhedra, now with a slice
#
/solid/G4Polyhedra 0 90 2 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 
/test/gridSizes 0 0 0 m
/test/errorFileName  log/polyhedra.d1.log
/test/run
/voxel/errorFileName log/polyhedrav.d1.log
/voxel/run
#
# --- polyhedra.d2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/polyhedra.d2.log
/test/run
#
# --- polyhedra.e1.log
# Build a much more complicated polyhedra, now with a thin slice
#
/solid/G4Polyhedra -1 2 2 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 
/test/gridSizes 0 0 0 m
/test/errorFileName  log/polyhedra.e1.log
/test/run
/voxel/errorFileName log/polyhedrav.e1.log
/voxel/run
#
# --- polyhedra.e2.log
# Up the ante and generate points on a grid
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/polyhedra.e2.log
/test/run
#
# --- polyhedra.f1.log
# One of my old favorites, with a few sharp turns
#
/solid/G4Polyhedra2 0 270 6 6 (-0.6,0.0,-1.0,0.5,0.5,1.0) (0.5,0.5,0.4,0.4,0.8,0.8) (0.6,0.6,1.0,1.0,1.0,1.1)
/test/gridSizes 0 0 0 m
/test/errorFileName  log/polyhedra.f1.log
/test/run
/voxel/errorFileName log/polyhedrav.f1.log
/voxel/run
#
/test/gridSizes 0.02 0.02 0.02 m
/test/errorFileName  log/polyhedra.f2.log
/test/run
#
exit
