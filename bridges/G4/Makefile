# ----------------------------------------------------------------
# GNUmakefile for Geant4 USolids brigde library.
# ----------------------------------------------------------------

name := G4USolids

ifndef G4INSTALL
  @echo "ERROR - No G4INSTALL defined for Geant4 Toolkit installation path !"
  @exit
endif

include $(G4INSTALL)/config/architecture.gmk

ifdef G4TARGET
  G4EXEC_BUILD = true

  # Override some variables for binmake.gmk.
  #
  INCFLAGS := -I$(G4BASE)/geometry/management/include \
              -I$(G4BASE)/geometry/volumes/include \
              -I$(G4BASE)/geometry/navigation/include \
              -I$(G4BASE)/geometry/solids/CSG/include \
              -I$(G4BASE)/geometry/solids/Boolean/include \
              -I$(G4BASE)/geometry/solids/specific/include \
              -I$(G4BASE)/global/management/include \
              -I$(G4BASE)/global/HEPRandom/include \
              -I$(G4BASE)/global/HEPNumerics/include \
              -I$(G4BASE)/global/HEPGeometry/include \
              -I$(G4BASE)/graphics_reps/include \
              -I../../include

  LDLIBS   := -lG4geomBoolean -lG4specsolids -lG4csg -lG4navigation \
              -lG4volumes -lG4geometrymng \
              -lG4graphics_reps -lG4hepnumerics\
              -lG4magneticfield -lG4intercoms\
              -lG4globman -lG4USolids ../../libUSolids.so

  include $(G4INSTALL)/config/binmake.gmk

else

  CPPFLAGS += -I$(G4BASE)/intercoms/include \
              -I$(G4BASE)/graphics_reps/include \
              -I$(G4BASE)/global/management/include \
              -I$(G4BASE)/global/HEPNumerics/include \
              -I$(G4BASE)/global/HEPRandom/include \
              -I$(G4BASE)/global/HEPGeometry/include \
              -I$(G4BASE)/geometry/volumes/include \
              -I$(G4BASE)/geometry/management/include \
              -I../../include

  include $(G4INSTALL)/config/common.gmk

endif

.PHONY: clean_all

clean_all:
	@for i in *.cc; do \
	(target=`basename $$i .cc`; \
	echo  "Removing $$target ... "; \
	$(MAKE) clean G4SKIP_DEPEND=1 G4TARGET=$$target); done
	@echo  "Wiping output file ... "
	@$(RM) -rf test.out
