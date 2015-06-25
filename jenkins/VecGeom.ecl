-project=VecGeom,getenv("PROJECT_ROOT")
-report_output={protobuf,getenv("PB_OUTPUT")}
-report_enabled={text,false},{protobuf,true}
-report_macros={protobuf,10}

-eval-file=GEANT_rules_config.ecl
-config=UCGP1.F1,+include_areas={{vecgeom,"!system","^(.*/)?VecGeom/"}}
-config=UCGP1.F1,+area_property={{vecgeom,0,-1,true}}
-config=UCGP1.F1,include_order={main,local,vecgeom,user_system}

-file_category={system,"^(/|<)"}

-locations={hide,{},system, "Disregard non-GEANT sources."}
-files={hide,system, "Disregard non-GEANT sources."}
-source_files={hide,system,"Disregard non-GEANT sources."}

-locations+={hide,{context},all, "Context locations are uninteresting."}
