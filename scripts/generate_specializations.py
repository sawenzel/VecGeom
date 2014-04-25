# Script to generate translation and rotation specializations of placed volumes,

rotation = [0x1B1, 0x18E, 0x076, 0x16A, 0x155, 0x0AD, 0x0DC, 0x0E3, 0x11B,
            0x0A1, 0x10A, 0x046, 0x062, 0x054, 0x111, 0x200]
translation = ["translation::kGeneric", "translation::kIdentity"]

output_string = """\
  if (trans_code == {:s} && rot_code == {:#05x}) {{
    return VolumeType::template Create<{:s}, {:#05x}>(
             logical_volume,
             transformation,
#ifdef VECGEOM_NVCC
             id,
#endif
             placement
           );
  }}\
"""

for r in rotation:
  for t in translation:
    print(output_string.format(t, r, t, r))