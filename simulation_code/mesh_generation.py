"""
Mesh generation for the 2D axisymmetric underwater acoustic coating model.

Geometry (axisymmetric about z-axis, r >= 0):
  - Water column on top (incident medium)
  - 10-layer rubber coating (some layers have hollow cylinders)
  - Rigid backing at the bottom
  - PML region surrounding the water to absorb outgoing waves

Uses gmsh Python API to build the geometry and mesh, then converts
to DOLFINx-compatible format.

Run with:  python mesh_generation.py          (creates mesh files)
           python mesh_generation.py --show    (opens gmsh GUI to inspect)
"""

import sys
import numpy as np

try:
    import gmsh
except ImportError:
    sys.exit(
        "ERROR: gmsh Python API not found.\n"
        "Install with:  pip install gmsh\n"
        "  (inside your WSL Ubuntu terminal)"
    )


def create_mesh(params_derived, mesh_file="coating_mesh.msh",
                water_height=0.5, pml_thickness=0.2,
                mesh_size_coating=0.002, mesh_size_water=0.01,
                show_gui=False):
    """
    Build 2D axisymmetric mesh of the underwater coating unit cell.

    Parameters
    ----------
    params_derived : dict from config.compute_derived()
    mesh_file      : output filename (.msh)
    water_height   : height of water column above coating (m)
    pml_thickness  : thickness of PML region around water (m)
    mesh_size_coating : target element size inside coating (m)
    mesh_size_water   : target element size in water (m)
    show_gui       : if True, open gmsh GUI before writing

    Returns
    -------
    dict with keys:
        "mesh_file"    : path to written .msh file
        "markers"      : dict of physical group tag -> name
    """
    R = params_derived["R_outer"]   # outer radius of unit cell
    d_m = params_derived["d_m"]     # list of 10 layer thicknesses (m)
    m_m = params_derived["m_m"]     # dict {layer_num: hollow_diameter_m}
    total_h = params_derived["total_thickness"]

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("underwater_coating")

    # ─── coordinate system ───
    # z-axis is the axis of symmetry (r = 0 is the left boundary)
    # z = 0  is the top of the coating (water-coating interface)
    # z < 0  goes into the coating towards rigid backing
    # z > 0  goes into the water column

    # ─── Physical tag IDs ───
    TAG_WATER = 1
    TAG_PML = 2
    TAG_COATING_SOLID = 3     # solid rubber layers (1,4,7,10)
    TAG_COATING_HOLLOW = 4    # hollow rubber layers (2,3,5,6,8,9) — rubber annulus
    TAG_CAVITY = 5            # air/vacuum inside hollows
    TAG_RIGID_BACK = 10       # boundary
    TAG_SYMMETRY = 11         # r = 0
    TAG_OUTER_WALL = 12       # r = R
    TAG_INCIDENT = 13         # top of water (where wave enters)
    TAG_FSI = 14              # fluid-structure interface

    # ─── Helper: add a rectangle in (r, z) ───
    surfaces = {}
    lines_fsi = []

    def add_rect(r0, z0, r1, z1, tag_name, lc):
        """Add a rectangle and return its surface tag."""
        p1 = gmsh.model.occ.addPoint(r0, z0, 0, lc)
        p2 = gmsh.model.occ.addPoint(r1, z0, 0, lc)
        p3 = gmsh.model.occ.addPoint(r1, z1, 0, lc)
        p4 = gmsh.model.occ.addPoint(r0, z1, 0, lc)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        s = gmsh.model.occ.addPlaneSurface([cl])
        return s

    # ─── 1. Water column ───
    water_surf = add_rect(0, 0, R, water_height, "water", mesh_size_water)

    # ─── 2. PML (surrounds water on top and right side) ───
    # Top PML
    pml_top = add_rect(0, water_height, R, water_height + pml_thickness,
                       "pml_top", mesh_size_water)
    # Right PML
    pml_right = add_rect(R, 0, R + pml_thickness, water_height + pml_thickness,
                         "pml_right", mesh_size_water)

    # ─── 3. Coating layers (z goes negative from 0) ───
    coating_surfs = []   # (layer_num, surface_tag, is_hollow)
    cavity_surfs = []    # (layer_num, surface_tag) for the air cavity inside hollow

    z_top = 0.0
    for i in range(10):
        layer_num = i + 1
        z_bot = z_top - d_m[i]

        if layer_num in m_m:
            # Hollow layer: rubber annulus + central cavity
            r_hole = m_m[layer_num] / 2.0   # radius of hollow

            # Rubber annulus (from r_hole to R)
            rubber_surf = add_rect(r_hole, z_bot, R, z_top,
                                   f"rubber_L{layer_num}", mesh_size_coating)
            coating_surfs.append((layer_num, rubber_surf, True))

            # Cavity (from 0 to r_hole)  — treated as air/fluid
            cav_surf = add_rect(0, z_bot, r_hole, z_top,
                                f"cavity_L{layer_num}", mesh_size_coating)
            cavity_surfs.append((layer_num, cav_surf))
        else:
            # Solid rubber layer (full width)
            solid_surf = add_rect(0, z_bot, R, z_top,
                                  f"solid_L{layer_num}", mesh_size_coating)
            coating_surfs.append((layer_num, solid_surf, False))

        z_top = z_bot

    # ─── 4. Synchronize and fragment for conforming mesh ───
    gmsh.model.occ.synchronize()

    # Fragment all surfaces so they share edges at interfaces
    all_surfs = (
        [(2, water_surf), (2, pml_top), (2, pml_right)]
        + [(2, s) for (_, s, _) in coating_surfs]
        + [(2, s) for (_, s) in cavity_surfs]
    )
    if len(all_surfs) > 1:
        gmsh.model.occ.fragment(all_surfs[:1], all_surfs[1:])
        gmsh.model.occ.synchronize()

    # ─── 5. Assign physical groups ───
    # After fragment, surface tags may change. We retrieve all surfaces
    # and classify them by their centre-of-mass position.
    all_surfaces = gmsh.model.getEntities(dim=2)

    water_tags = []
    pml_tags = []
    solid_tags = []
    hollow_tags = []
    cavity_tags = []

    for dim, tag in all_surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        r_min, z_min, _, r_max, z_max, _ = bb
        r_c = (r_min + r_max) / 2
        z_c = (z_min + z_max) / 2

        # Classify
        if z_min >= water_height - 1e-8:
            # PML top
            pml_tags.append(tag)
        elif r_min >= R - 1e-8:
            # PML right
            pml_tags.append(tag)
        elif z_min >= -1e-8 and z_max <= water_height + 1e-8 and r_max <= R + 1e-8:
            # Water
            water_tags.append(tag)
        elif z_max <= 1e-8:
            # Inside coating region
            # Check if it's a cavity (r_max < some hollow radius) or rubber
            # We determine by layer from z-position
            z_acc = 0.0
            found = False
            for li in range(10):
                ln = li + 1
                z_acc -= d_m[li]
                if z_min >= z_acc - 1e-8 and z_max <= (z_acc + d_m[li]) + 1e-8:
                    if ln in m_m:
                        r_hole = m_m[ln] / 2.0
                        if r_max <= r_hole + 1e-6:
                            cavity_tags.append(tag)
                        else:
                            hollow_tags.append(tag)
                    else:
                        solid_tags.append(tag)
                    found = True
                    break
            if not found:
                # Fallback: treat as solid coating
                solid_tags.append(tag)

    # Create physical groups
    if water_tags:
        gmsh.model.addPhysicalGroup(2, water_tags, TAG_WATER)
        gmsh.model.setPhysicalName(2, TAG_WATER, "water")
    if pml_tags:
        gmsh.model.addPhysicalGroup(2, pml_tags, TAG_PML)
        gmsh.model.setPhysicalName(2, TAG_PML, "pml")
    if solid_tags:
        gmsh.model.addPhysicalGroup(2, solid_tags, TAG_COATING_SOLID)
        gmsh.model.setPhysicalName(2, TAG_COATING_SOLID, "coating_solid")
    if hollow_tags:
        gmsh.model.addPhysicalGroup(2, hollow_tags, TAG_COATING_HOLLOW)
        gmsh.model.setPhysicalName(2, TAG_COATING_HOLLOW, "coating_hollow")
    if cavity_tags:
        gmsh.model.addPhysicalGroup(2, cavity_tags, TAG_CAVITY)
        gmsh.model.setPhysicalName(2, TAG_CAVITY, "cavity")

    # ─── 6. Boundary physical groups ───
    all_curves = gmsh.model.getEntities(dim=1)
    rigid_back_curves = []
    symmetry_curves = []
    outer_wall_curves = []
    incident_curves = []

    z_bottom = -sum(d_m)
    for dim, tag in all_curves:
        bb = gmsh.model.getBoundingBox(dim, tag)
        r_min, z_min, _, r_max, z_max, _ = bb
        length_r = r_max - r_min
        length_z = z_max - z_min

        # Rigid backing: horizontal line at z = z_bottom
        if abs(z_min - z_bottom) < 1e-8 and abs(z_max - z_bottom) < 1e-8:
            rigid_back_curves.append(tag)
        # Symmetry axis: vertical line at r = 0
        elif abs(r_min) < 1e-8 and abs(r_max) < 1e-8:
            symmetry_curves.append(tag)
        # Outer wall: vertical line at r = R (inside coating region only)
        elif abs(r_min - R) < 1e-8 and abs(r_max - R) < 1e-8 and z_max <= 1e-8:
            outer_wall_curves.append(tag)
        # Incident surface: horizontal at top of PML
        elif abs(z_min - (water_height + pml_thickness)) < 1e-8 and abs(z_max - (water_height + pml_thickness)) < 1e-8:
            incident_curves.append(tag)

    if rigid_back_curves:
        gmsh.model.addPhysicalGroup(1, rigid_back_curves, TAG_RIGID_BACK)
        gmsh.model.setPhysicalName(1, TAG_RIGID_BACK, "rigid_backing")
    if symmetry_curves:
        gmsh.model.addPhysicalGroup(1, symmetry_curves, TAG_SYMMETRY)
        gmsh.model.setPhysicalName(1, TAG_SYMMETRY, "symmetry_axis")
    if outer_wall_curves:
        gmsh.model.addPhysicalGroup(1, outer_wall_curves, TAG_OUTER_WALL)
        gmsh.model.setPhysicalName(1, TAG_OUTER_WALL, "outer_wall")
    if incident_curves:
        gmsh.model.addPhysicalGroup(1, incident_curves, TAG_INCIDENT)
        gmsh.model.setPhysicalName(1, TAG_INCIDENT, "incident_surface")

    # ─── 7. Generate mesh ───
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(mesh_file)
    print(f"Mesh written to: {mesh_file}")

    if show_gui:
        gmsh.fltk.run()

    markers = {
        TAG_WATER: "water",
        TAG_PML: "pml",
        TAG_COATING_SOLID: "coating_solid",
        TAG_COATING_HOLLOW: "coating_hollow",
        TAG_CAVITY: "cavity",
        TAG_RIGID_BACK: "rigid_backing",
        TAG_SYMMETRY: "symmetry_axis",
        TAG_OUTER_WALL: "outer_wall",
        TAG_INCIDENT: "incident_surface",
    }

    gmsh.finalize()

    return {"mesh_file": mesh_file, "markers": markers}


# ─── CLI entry point ───
if __name__ == "__main__":
    from config import get_base_case_params, compute_derived

    params = compute_derived(get_base_case_params())
    show = "--show" in sys.argv
    result = create_mesh(params, show_gui=show)
    print("Physical group markers:", result["markers"])
