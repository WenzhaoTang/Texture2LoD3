# CityGML to OBJ Converter

Convert 3D CityGML (.gml) files to Wavefront OBJ (.obj) format, preserving geometry and optional semantics.

## Prerequisites
  - Python 3.6 or later
  - Install dependencies:
    ```bash
    pip install -r obj_extraction/CityGML2OBJv2/requirements.txt
    ```

## Usage
Run the converter from the repository root:
```bash
python obj_extraction/CityGML2OBJv2/CityGML2OBJs.py \
  -i <input_folder> \
  -o <output_folder> \
  [-s 1]  # semantic separation (default 0)
  [-g 1]  # separate per building (default 0)
  [-v 1]  # validate polygons (default 0)
  [-p 1]  # preserve polygons (skip triangulation)
  [-t 1]  # translate coordinates (min vertex to origin)
```
Use `-h` or `--help` to see all available options.

## Example
Convert CityGML files in `data/citygml` to OBJ in `output/obj` with semantic separation:
```bash
python obj_extraction/CityGML2OBJv2/CityGML2OBJs.py \
  -i data/citygml \
  -o output/obj \
  -s 1
```
The resulting `.obj` and `.mtl` files will be in `output/obj`.

## License
See `obj_extraction/CityGML2OBJv2/LICENSE` for license details.