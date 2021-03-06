#
# GEANT4 SBT Script to test G4TessellatedSolid
# I.Hrivnacova, IPN Orsay 28/01/2008 

#
/test/maxPoints 1000
#
# --- tessellatedSolid.a1.log
# Extruded solid with triangular polygon
#
/solid/G4TessellatedSolid 2 (-0.3,-0.3,-0.3,0.3,-0.3,0.3) (0.0,0.3,-0.3,0.0,0.3,0.3) (0.3,-0.3,-0.3,-0.3,-0.3,0.3) 3 (0.0,0.3,-0.3,0.3,-0.3,-0.3,-0.3,-0.3,-0.3) (-0.3,-0.3,-0.3,0.0,0.3,-0.3,0.3,-0.3,-0.3) (-0.3,-0.3,0.3,0.0,0.3,0.3,0.3,-0.3,0.3) (0.0,0.3,0.3,0.3,-0.3,0.3,-0.3,-0.3,0.3)
/test/gridSizes 0.1 0.1 0.1 m
/test/errorFileName  log/tessellatedSolid.a1.log
/test/run
/voxel/errorFileName log/tessellatedSolidv.a1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tessellatedSolid.a2.log
/test/run
#

# --- tessellatedSolid.b1.log
# Box defined as Extruded solid interpreted as Tessellated solid
#
/solid/G4TessellatedSolid2 4 (-0.3,-0.3,0.3,0.3) (-0.3,0.3,0.3,-0.3) 2 (-0.3,0.3) (0.0,0.0) (0.0,0.0) (1.0,1.0)
/test/gridSizes 0.1 0.1 0.1 m
/test/errorFileName  log/tessellatedSolid.b1.log
/test/run
/voxel/errorFileName log/tessellatedSolidv.b1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tessellatedSolid.b2.log
/test/run
#
# --- tessellatedSolid.c1.log
# Extruded solid with 4 z-sections interpreted as Tessellated solid
#
/solid/G4TessellatedSolid2 8 (-0.3,-0.3,0.3,0.3,0.15,0.15,-0.15,-0.15) (-0.3,0.3,0.3,-0.3,-0.3,0.15,0.15,-0.3) 4 (-0.4,0.1,0.15,0.4) (-0.2,0.0,0.0,0.2) (0.1,0.0,0.0,0.2) (1.5,0.5,0.7,0.9)
/test/gridSizes 0.1 0.1 0.1 m
/test/errorFileName  log/tessellatedSolid.c1.log
/test/run
/voxel/errorFileName log/tessellatedSolidv.c1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tessellatedSolid.c2.log
/test/run
#
# --- tessellatedSolid.d1.log
# Another extruded solid, where polygon decomposition was failing,
# interpreted as Tessellated solid
# in Geant4 9.1
#
/solid/G4TessellatedSolid2 8 (-0.2,-0.2,0.1,0.1,0.2,0.2,-0.1,-0.1) (0.1,0.25,0.25,-0.1,-0.1,-0.25,-0.25,0.1) 2 (-0.2,0.2) (0.0,0.0) (0.0,0.0) (1.0,1.0)
/test/gridSizes 0.1 0.1 0.1 m
/test/errorFileName  log/tessellatedSolid.d1.log
/test/run
/voxel/errorFileName log/tessellatedSolidv.d1.log
/voxel/run
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/tessellatedSolid.d2.log
/test/run
#
exit
