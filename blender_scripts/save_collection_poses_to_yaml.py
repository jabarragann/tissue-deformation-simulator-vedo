from pathlib import Path

import bpy

# ---------------- CONFIG ----------------
COLLECTION_NAMES = [
    "spheres_before_deformation",
    "spheres_after_deformation",
    "landmarks_after",
    "landmarks_before",
]

OUTPUT_PATH = Path(bpy.path.abspath("//object_positions.yaml"))
FLOAT_FMT = "{:12.6f}"  # width=12, precision=6
# ---------------------------------------


def objects_in_collection_recursive(collection):
    objs = list(collection.objects)
    for child in collection.children:
        objs.extend(objects_in_collection_recursive(child))
    return objs


yaml_lines = []

for coll_name in COLLECTION_NAMES:
    coll = bpy.data.collections.get(coll_name)

    if coll is None:
        print(f"Warning: Collection '{coll_name}' not found, skipping")
        continue

    print(f"Saving collection: {coll_name}")
    yaml_lines.append(f"{coll_name}:")

    for obj in objects_in_collection_recursive(coll):
        loc = obj.matrix_world.translation

        x = FLOAT_FMT.format(loc.x)
        y = FLOAT_FMT.format(loc.y)
        z = FLOAT_FMT.format(loc.z)

        line = f"    {obj.name}: [{x}, {y}, {z}]"
        print(line)
        yaml_lines.append(line)

    yaml_lines.append("")  # blank line between collections


OUTPUT_PATH.write_text("\n".join(yaml_lines).rstrip() + "\n", encoding="utf-8")

print(f"YAML written to: {OUTPUT_PATH}")
