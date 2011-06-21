void run()
{
// Run a solid test
   gSystem->Load("libGeom.so");
   gSystem->Load("libUSolids.so");
   gSystem->Load("libUBridges.so");
   TString incpath = Form("-I%s/include -I%s/bridges/TGeo", 
               gSystem->ExpandPathName("$FULLPATH"), gSystem->ExpandPathName("$FULLPATH"));
   gSystem->AddIncludePath(incpath);
   printf("Include path: %s\n", gSystem->GetIncludePath());
   gROOT->LoadMacro("TestBox.C+g");
}   
