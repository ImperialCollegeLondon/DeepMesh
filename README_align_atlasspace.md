
## Overview
The ''align_atlasspace.py'' processes 4D cardiac data by applying **Rigid Registration** to align subject meshes with an atlas and transforem all the subjects from subject-space to atlas-space. It utilizes libraries such as **VTK** and  **IRTK** for mesh handling and **Nibabel** for NIfTI file processing.

## Features
- Reads and processes 4D cardiac mesh data.
- Aligns (Rigid Regiateration) subject meshes to a common atlas space.
- Outputs aligned mesh files in VTK format.

## Processing
Use the following command to process your data:
```bash
python align_atlasspace.py --UKBB_DIR /path/to/ukbb --MESH_DIR /path/to/mesh --ATLAS_DIR /path/to/atlas --OUTPUT_DIR /path/to/output --SECTION LVmyo --N_FRAME 50

```

## Prerequisites
Before running the script, make sure you have the following Python libraries installed:

```bash
pip install nibabel vtk tqdm numpy irtk

```

