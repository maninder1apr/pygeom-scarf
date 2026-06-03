"""
SCARF PEN Enclosure Geometry
============================

This module defines PEN enclosure geometries **specific to the SCARF
experimental setup**. The enclosures are designed for the two
non-enriched test detectors used in SCARF:

    - BEGe type (e.g. B00002C)
    - ICPC type (e.g. V07302A)

These are NOT enriched LEGEND detectors and this implementation is
NOT intended to be general purpose.

The hardcoded dimensions (wall thickness, flange thickness, cap
thickness) are properties of the SCARF PEN hardware design taken
from the mechanical drawings — not the detector geometry, which is
read from detector metadata at runtime.

This is an ad-hoc implementation to test whether approximate
geometries built from boolean unions reproduce the expected K-42
survival fraction in SCARF. A more detailed design can be loaded
directly from an STL file of the actual CAD model, which would
replace the boolean union construction entirely.

Shapes available
----------------
flat        — baseline cylindrical shell (current design)
ribbed      — thin base shell + horizontal ribs
slotted     — shell with LAr-filled rectangular slots
layered     — multiple concentric shells with LAr gaps
honeycomb   — azimuthal segments alternating with LAr gaps
fins        — thin base shell + inward-pointing radial fins

All shapes are conformal — the inner surface hugs the HPGe detector
with minimal clearance to maximize particle interception.

Mass and S/V utilities
----------------------
compute_pen_mass_g()   — total PEN mass in grams
compute_pen_sv_ratio() — surface-to-volume ratio in 1/mm
"""

from __future__ import annotations

import math

from pyg4ometry.geant4 import Registry, solid

TWO_PI = 2 * math.pi
COPLANAR_SEPARATION_MM = 0.01
PEN_DENSITY_G_CM3 = 1.36  # measured PEN density


# ─────────────────────────────────────────────────────────────────────────────
# Detector parameters
# ─────────────────────────────────────────────────────────────────────────────

PEN_ENCLOSURES = {
    "bege": {
        # HPGe dimensions (from detector metadata)
        "det_r_mm": 37.0,
        "det_h_mm": 32.0,
        # enclosure hardware dimensions (from SCARF mechanical drawings)
        "body_inner_r_mm": 37.0,  # hugs HPGe surface
        "body_outer_r_mm": 38.5,  # 1.5mm wall baseline
        "body_h_mm": 33.0,
        "flange_outer_r_mm": 44.5,
        "flange_inner_r_mm": 33.75,
        "flange_t_mm": 1.5,
        "cap_t_mm": 1.5,
        "z_offset_mm": 16.0,
        # shape parameters
        "n_shells": 3,
        "shell_t_mm": 2.0,
        "gap_t_mm": 3.0,
        "n_ribs": 6,
        "rib_t_mm": 3.0,
        "rib_h_mm": 2.0,
        "rib_spacing_mm": 5.0,
        "n_slots": 8,
        "slot_w_mm": 8.0,
        "slot_h_mm": 20.0,
        "cell_size_mm": 8.0,
        "n_fins": 16,
        "fin_l_mm": 4.0,  # must be < gap between HPGe and PEN
        "fin_t_mm": 1.0,
    },
    "icpc": {
        # HPGe dimensions
        "det_r_mm": 40.0,
        "det_h_mm": 65.0,
        # enclosure hardware dimensions — 1.5mm clearance from HPGe surface
        "body_inner_r_mm": 41.5,  # det_r + 1.5mm clearance
        "body_outer_r_mm": 43.0,  # inner + 1.5mm wall
        "body_h_mm": 65.0,  # body shorter than HPGe, caps cover rest
        "flange_outer_r_mm": 46.0,
        "flange_inner_r_mm": 28.0,
        "flange_t_mm": 1.5,
        "cap_t_mm": 1.5,
        "borehole_r_mm": 5.0,
        "borehole_h_mm": 32.0,
        "z_offset_mm": 32.5,
        # shape parameters — tuned for ICPC (taller, larger radius)
        "n_shells": 2,
        "shell_t_mm": 1.5,
        "gap_t_mm": 2.0,
        "n_ribs": 8,
        "rib_t_mm": 3.0,
        "rib_h_mm": 2.0,
        "rib_spacing_mm": 8.0,
        "n_slots": 8,
        "slot_w_mm": 8.0,
        "slot_h_mm": 40.0,
        "cell_size_mm": 6.0,
        "n_fins": 20,
        "fin_l_mm": 3.0,  # conservative — stays within clearance
        "fin_t_mm": 1.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper — borehole plug
# ─────────────────────────────────────────────────────────────────────────────


def _add_borehole_plug(
    name: str,
    shell: solid.Union,
    body_half_height: float,
    flange_t_mm: float,
    cap_t_mm: float,
    borehole_r_mm: float,
    borehole_h_mm: float,
    registry: Registry,
    det_h_mm: float = 0.0,
) -> solid.Union:
    borehole_half_height = (borehole_h_mm - COPLANAR_SEPARATION_MM) / 2.0
    plug = solid.Tubs(
        f"{name}_borehole_plug",
        0,
        borehole_r_mm - COPLANAR_SEPARATION_MM,
        borehole_half_height,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    if det_h_mm > 0:
        # plug spans from crystal bottom (-det_h/2) upward by borehole_h_mm
        plug_z = -det_h_mm / 2.0 + borehole_half_height + COPLANAR_SEPARATION_MM
    else:
        plug_z = -body_half_height + borehole_half_height + COPLANAR_SEPARATION_MM

    return solid.Union(
        f"{name}_u_borehole",
        shell,
        plug,
        [[0, 0, 0], [0, 0, plug_z]],
        registry,
    )


def _add_caps_and_flanges(
    name: str,
    shell: solid.Union,
    body_half_height: float,
    r_outer: float,
    cap_t_mm: float,
    flange_t_mm: float,
    flange_inner_r_mm: float,
    flange_outer_r_mm: float,
    registry: Registry,
    r_inner: float = 0.0,
) -> solid.Union:
    """Add top/bottom caps and flanges to an existing shell union."""
    cap_half = cap_t_mm / 2.0
    flange_half = flange_t_mm / 2.0

    cap_top = solid.Tubs(
        f"{name}_cap_top", r_inner, r_outer, cap_half, 0, TWO_PI, registry=registry, lunit="mm"
    )
    cap_bot = solid.Tubs(
        f"{name}_cap_bot", r_inner, r_outer, cap_half, 0, TWO_PI, registry=registry, lunit="mm"
    )
    flange_top = solid.Tubs(
        f"{name}_flange_top",
        flange_inner_r_mm,
        flange_outer_r_mm,
        flange_half,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )
    flange_bot = solid.Tubs(
        f"{name}_flange_bot",
        flange_inner_r_mm,
        flange_outer_r_mm,
        flange_half,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    shell = solid.Union(
        f"{name}_u_cap_top",
        shell,
        cap_top,
        [[0, 0, 0], [0, 0, body_half_height + cap_half + COPLANAR_SEPARATION_MM]],
        registry,
    )
    shell = solid.Union(
        f"{name}_u_cap_bot",
        shell,
        cap_bot,
        [[0, 0, 0], [0, 0, -body_half_height - cap_half - COPLANAR_SEPARATION_MM]],
        registry,
    )
    shell = solid.Union(
        f"{name}_u_flange_top",
        shell,
        flange_top,
        [[0, 0, 0], [0, 0, body_half_height + cap_t_mm + flange_half + COPLANAR_SEPARATION_MM]],
        registry,
    )
    return solid.Union(
        f"{name}_u_flange_bot",
        shell,
        flange_bot,
        [[0, 0, 0], [0, 0, -body_half_height - cap_t_mm - flange_half - COPLANAR_SEPARATION_MM]],
        registry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape 1 — flat conformal shell (baseline)
# ─────────────────────────────────────────────────────────────────────────────


def build_pen_polycone(
    name: str,
    *,
    body_outer_r_mm: float,
    body_inner_r_mm: float,
    body_h_mm: float,
    flange_outer_r_mm: float,
    flange_inner_r_mm: float,
    flange_t_mm: float,
    cap_t_mm: float,
    det_h_mm: float = 0.0,  # <-- add this
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Baseline flat conformal cylindrical shell hugging HPGe surface."""
    body_half_height = body_h_mm / 2.0

    body = solid.Tubs(
        f"{name}_body", body_inner_r_mm, body_outer_r_mm, body_h_mm, 0, TWO_PI, registry=registry, lunit="mm"
    )

    return _add_caps_and_flanges(
        name,
        body,
        body_half_height,
        body_outer_r_mm,
        cap_t_mm,
        flange_t_mm,
        flange_inner_r_mm,
        flange_outer_r_mm,
        registry,
        r_inner=body_inner_r_mm,
    )

    # if borehole_r_mm is not None and borehole_h_mm is not None:
    #    shell = _add_borehole_plug(
    #        name, shell, body_half_height,
    #        flange_t_mm, cap_t_mm,
    #        borehole_r_mm, borehole_h_mm, registry,
    #        det_h_mm=det_h_mm,    # <-- add this
    #    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape 2 — conformal ribbed shell
# ─────────────────────────────────────────────────────────────────────────────


def build_pen_ribbed_shell(
    name: str,
    *,
    body_inner_r_mm: float,
    body_outer_r_mm: float,
    body_h_mm: float,
    flange_outer_r_mm: float,
    flange_inner_r_mm: float,
    flange_t_mm: float,
    cap_t_mm: float,
    rib_t_mm: float = 3.0,
    rib_h_mm: float = 2.0,
    rib_spacing_mm: float = 5.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal thin base shell with horizontal annular ribs.

    The 0.5mm base shell hugs HPGe. Ribs extend outward adding PEN
    material at regular intervals. LAr fills the gaps between ribs
    providing additional scintillation signal.
    """
    body_half_height = body_h_mm / 2.0
    base_wall = 0.5  # thin base hugging HPGe

    base = solid.Tubs(
        f"{name}_base",
        body_inner_r_mm,
        body_inner_r_mm + base_wall,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    n_ribs = max(1, int(body_h_mm / rib_spacing_mm))
    shell = base
    for i in range(n_ribs):
        z_pos = -body_half_height + (i + 0.5) * rib_spacing_mm
        rib = solid.Tubs(
            f"{name}_rib_{i}",
            body_inner_r_mm,
            body_inner_r_mm + base_wall + rib_t_mm,
            rib_h_mm / 2.0,
            0,
            TWO_PI,
            registry=registry,
            lunit="mm",
        )
        shell = solid.Union(f"{name}_u_rib_{i}", shell, rib, [[0, 0, 0], [0, 0, z_pos]], registry)

    r_outer = body_inner_r_mm + base_wall + rib_t_mm
    return _add_caps_and_flanges(
        name,
        shell,
        body_half_height,
        r_outer,
        cap_t_mm,
        flange_t_mm,
        flange_inner_r_mm,
        flange_outer_r_mm,
        registry,
        r_inner=body_inner_r_mm,
    )

    # if borehole_r_mm is not None and borehole_h_mm is not None:
    #    shell = _add_borehole_plug(
    #        name, shell, body_half_height,
    #        flange_t_mm, cap_t_mm,
    #        borehole_r_mm, borehole_h_mm, registry
    #    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape 3 — conformal slotted shell
# ─────────────────────────────────────────────────────────────────────────────


def build_pen_slotted_shell(
    name: str,
    *,
    body_inner_r_mm: float,
    body_outer_r_mm: float,
    body_h_mm: float,
    flange_outer_r_mm: float,
    flange_inner_r_mm: float,
    flange_t_mm: float,
    cap_t_mm: float,
    n_slots: int = 8,
    slot_w_mm: float = 8.0,
    slot_h_mm: float = 20.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal shell with rectangular LAr-filled slots.

    Slots cut through the shell wall allow LAr to penetrate,
    giving double veto signal — PEN scintillation at slot edges
    and LAr scintillation inside slots.
    """
    body_half_height = body_h_mm / 2.0
    wall_t = body_outer_r_mm - body_inner_r_mm

    base = solid.Tubs(
        f"{name}_base", body_inner_r_mm, body_outer_r_mm, body_h_mm, 0, TWO_PI, registry=registry, lunit="mm"
    )

    shell = base
    for i in range(n_slots):
        angle = i * TWO_PI / n_slots
        slot = solid.Box(
            f"{name}_slot_{i}", slot_w_mm, wall_t + 2.0, slot_h_mm, registry=registry, lunit="mm"
        )
        x = (body_inner_r_mm + wall_t / 2.0) * math.cos(angle)
        y = (body_inner_r_mm + wall_t / 2.0) * math.sin(angle)
        shell = solid.Subtraction(f"{name}_slot_cut_{i}", shell, slot, [[0, 0, angle], [x, y, 0]], registry)

    return _add_caps_and_flanges(
        name,
        shell,
        body_half_height,
        body_outer_r_mm,
        cap_t_mm,
        flange_t_mm,
        flange_inner_r_mm,
        flange_outer_r_mm,
        registry,
        r_inner=body_inner_r_mm,
    )

    # if borehole_r_mm is not None and borehole_h_mm is not None:
    #    shell = _add_borehole_plug(
    #        name, shell, body_half_height,
    #        flange_t_mm, cap_t_mm,
    #        borehole_r_mm, borehole_h_mm, registry
    #    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape 4 — conformal layered shells
# ─────────────────────────────────────────────────────────────────────────────


def build_pen_layered_shells(
    name: str,
    *,
    body_inner_r_mm: float,
    body_h_mm: float,
    flange_outer_r_mm: float,
    flange_inner_r_mm: float,
    flange_t_mm: float,
    cap_t_mm: float,
    n_shells: int = 3,
    shell_t_mm: float = 2.0,
    gap_t_mm: float = 3.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal concentric multi-shell PEN enclosure with LAr gaps.

    Multiple thin shells maximize particle energy loss while keeping
    individual shell thickness small for good light escape. LAr gaps
    between shells provide additional scintillation signal.
    """
    if n_shells < 1:
        msg = f"n_shells must be >= 1, got {n_shells}"
        raise ValueError(msg)
    if shell_t_mm <= 0:
        msg = f"shell_t_mm must be > 0, got {shell_t_mm}"
        raise ValueError(msg)
    if gap_t_mm <= 0:
        msg = f"gap_t_mm must be > 0, got {gap_t_mm}"
        raise ValueError(msg)

    body_half_height = body_h_mm / 2.0

    shell = None
    for i in range(n_shells):
        r_in = body_inner_r_mm + i * (shell_t_mm + gap_t_mm)
        r_out = r_in + shell_t_mm
        tub = solid.Tubs(
            f"{name}_shell_{i}", r_in, r_out, body_h_mm, 0, TWO_PI, registry=registry, lunit="mm"
        )
        if shell is None:
            shell = tub
        else:
            shell = solid.Union(f"{name}_u_shell_{i}", shell, tub, [[0, 0, 0], [0, 0, 0]], registry)

    r_outer = body_inner_r_mm + n_shells * shell_t_mm + (n_shells - 1) * gap_t_mm

    return _add_caps_and_flanges(
        name,
        shell,
        body_half_height,
        r_outer,
        cap_t_mm,
        flange_t_mm,
        flange_inner_r_mm,
        flange_outer_r_mm,
        registry,
        r_inner=body_inner_r_mm,
    )

    # if borehole_r_mm is not None and borehole_h_mm is not None:
    #    shell = _add_borehole_plug(
    #        name, shell, body_half_height,
    #        flange_t_mm, cap_t_mm,
    #        borehole_r_mm, borehole_h_mm, registry
    #    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape 5 — conformal honeycomb
# ─────────────────────────────────────────────────────────────────────────────
def build_pen_honeycomb(
    name: str,
    *,
    body_inner_r_mm: float,
    body_outer_r_mm: float,
    body_h_mm: float,
    flange_outer_r_mm: float,
    flange_inner_r_mm: float,
    flange_t_mm: float,
    cap_t_mm: float,
    cell_size_mm: float = 4.0,
    shell_t_mm: float = 1.0,
    n_layers: int = 2,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal honeycomb PEN enclosure with continuous hexagonal lattice.

    Built by subtracting hexagonal prisms from a solid PEN annular
    cylinder. Creates a true honeycomb lattice like a beehive with
    continuous shared walls between cells. LAr fills the hollow cells.

    Parameters
    ----------
    cell_size_mm
        Hexagonal cell radius in mm. Default 4mm.
    shell_t_mm
        PEN wall thickness in mm. Default 1mm.
    n_layers
        Number of hexagonal cell layers radially. Default 2.
    """
    import math as _math

    body_half_height = body_h_mm / 2.0
    cell_r = cell_size_mm
    wall_t = shell_t_mm
    r_inner = body_inner_r_mm
    r_outer = r_inner + n_layers * (2 * cell_r + wall_t)

    dx = (2 * cell_r + wall_t) * _math.cos(_math.radians(30))
    dy = 2 * cell_r + wall_t

    def _hex_verts(r):
        return [
            [r * _math.cos(_math.radians(30 + 60 * i)), r * _math.sin(_math.radians(30 + 60 * i))]
            for i in range(6)
        ]

    # generate cell centers
    centers = []
    rows = int((r_outer + cell_r) / dy) + 3
    cols = int((r_outer + cell_r) / dx) + 3
    for row in range(-rows, rows + 1):
        for col in range(-cols, cols + 1):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy * _math.sqrt(3) / 2
            d = _math.sqrt(x**2 + y**2)
            if r_inner < d < r_outer:
                centers.append((x, y))

    # solid PEN annular cylinder
    pen = solid.Tubs(
        f"{name}_outer",
        r_inner,
        r_outer,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    # subtract hexagonal cells
    void_r = cell_r - wall_t / 2.0
    verts = _hex_verts(void_r)

    for i, (cx, cy) in enumerate(centers):
        cell = solid.ExtrudedSolid(
            f"{name}_cell_{i}",
            verts,
            [[-body_h_mm / 2 - 1, [0, 0], 1], [body_h_mm / 2 + 1, [0, 0], 1]],
            registry,
        )
        pen = solid.Subtraction(
            f"{name}_sub_{i}",
            pen,
            cell,
            [[0, 0, 0], [cx, cy, 0]],
            registry,
        )

    return _add_caps_and_flanges(
        name,
        pen,
        body_half_height,
        r_outer,
        cap_t_mm,
        flange_t_mm,
        flange_inner_r_mm,
        flange_outer_r_mm,
        registry,
        r_inner=body_inner_r_mm,
    )

    # if borehole_r_mm is not None and borehole_h_mm is not None:
    #    shell = _add_borehole_plug(
    #        name, shell, body_half_height,
    #        flange_t_mm, cap_t_mm,
    #        borehole_r_mm, borehole_h_mm, registry
    #    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape 6 — conformal fins
# ─────────────────────────────────────────────────────────────────────────────


def build_pen_fins(
    name: str,
    *,
    body_inner_r_mm: float,
    body_outer_r_mm: float,
    body_h_mm: float,
    flange_outer_r_mm: float,
    flange_inner_r_mm: float,
    flange_t_mm: float,
    cap_t_mm: float,
    n_fins: int = 16,
    fin_l_mm: float = 6.0,
    fin_t_mm: float = 1.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal thin base shell with inward-pointing radial fins.

    A thin base shell hugs the HPGe outer surface. Thin fins point
    radially INWARD toward the HPGe surface. Any particle exiting
    HPGe must cross a fin immediately. High S/V ratio, very low mass.

    Parameters
    ----------
    n_fins
        Number of fins around circumference. Default 16.
    fin_l_mm
        Length of each fin in mm (radial depth). Default 6mm.
    fin_t_mm
        Thickness of each fin in mm. Default 1mm.
    """
    import math as _math

    body_half_height = body_h_mm / 2.0
    base_wall = 1.0  # thin base shell thickness

    # thin base shell on outer surface of HPGe
    base = solid.Tubs(
        f"{name}_base",
        body_inner_r_mm,
        body_inner_r_mm + base_wall,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    shell = base

    # fins point inward — from base shell toward HPGe center
    for i in range(n_fins):
        angle = i * TWO_PI / n_fins

        # fin center: at base shell inner surface, pointing inward
        # fin tip points toward HPGe, fin base connects to shell
        fin_center_r = body_inner_r_mm + fin_l_mm / 2.0
        fx = fin_center_r * _math.cos(angle)
        fy = fin_center_r * _math.sin(angle)

        fin = solid.Box(
            f"{name}_fin_{i}",
            fin_t_mm,
            fin_l_mm,
            body_h_mm,
            registry=registry,
            lunit="mm",
        )

        shell = solid.Union(
            f"{name}_u_fin_{i}",
            shell,
            fin,
            [[0, 0, angle], [fx, fy, 0]],
            registry,
        )

    r_outer = body_inner_r_mm + base_wall

    return _add_caps_and_flanges(
        name,
        shell,
        body_half_height,
        r_outer,
        cap_t_mm,
        flange_t_mm,
        flange_inner_r_mm,
        flange_outer_r_mm,
        registry,
        r_inner=body_inner_r_mm,
    )

    # if borehole_r_mm is not None and borehole_h_mm is not None:
    #    shell = _add_borehole_plug(
    #        name, shell, body_half_height,
    #        flange_t_mm, cap_t_mm,
    #        borehole_r_mm, borehole_h_mm, registry
    #    )


# ─────────────────────────────────────────────────────────────────────────────
# Mass and S/V calculators
# ─────────────────────────────────────────────────────────────────────────────


def compute_pen_mass_g(
    shape: str,
    detector_type: str,
    density_g_cm3: float = PEN_DENSITY_G_CM3,
) -> float:
    """Compute approximate total PEN mass in grams for a given shape.

    Parameters
    ----------
    shape
        One of the keys in PEN_SHAPE_BUILDERS.
    detector_type
        One of 'bege', 'icpc'.
    density_g_cm3
        PEN density in g/cm³. Default 1.36.

    Returns
    -------
    Mass in grams.
    """
    p = PEN_ENCLOSURES[detector_type]
    r_in = p["body_inner_r_mm"]
    h = p["body_h_mm"]

    if shape == "flat":
        r_out = p["body_outer_r_mm"]
        vol = math.pi * (r_out**2 - r_in**2) * h

    elif shape == "ribbed":
        base_wall = 0.5
        n_ribs = max(1, int(h / p["rib_spacing_mm"]))
        vol_base = math.pi * ((r_in + base_wall) ** 2 - r_in**2) * h
        r_rib = r_in + base_wall + p["rib_t_mm"]
        vol_ribs = n_ribs * math.pi * (r_rib**2 - r_in**2) * p["rib_h_mm"]
        vol = vol_base + vol_ribs

    elif shape == "slotted":
        r_out = p["body_outer_r_mm"]
        vol_shell = math.pi * (r_out**2 - r_in**2) * h
        wall_t = r_out - r_in
        vol_slot = p["n_slots"] * p["slot_w_mm"] * wall_t * p["slot_h_mm"]
        vol = vol_shell - vol_slot

    elif shape == "layered":
        vol = 0.0
        for i in range(p["n_shells"]):
            r_s_in = r_in + i * (p["shell_t_mm"] + p["gap_t_mm"])
            r_s_out = r_s_in + p["shell_t_mm"]
            vol += math.pi * (r_s_out**2 - r_s_in**2) * h

    elif shape == "honeycomb":
        r_out = p["body_outer_r_mm"]
        n_cells = max(6, int(TWO_PI * r_in / p["cell_size_mm"]))
        fill_fraction = 1.0 - (p["shell_t_mm"] / r_in) / (TWO_PI / n_cells)
        vol = math.pi * (r_out**2 - r_in**2) * h * fill_fraction

    elif shape == "fins":
        base_wall = 0.5
        vol_base = math.pi * ((r_in + base_wall) ** 2 - r_in**2) * h
        vol_fins = p["n_fins"] * p["fin_t_mm"] * p["fin_l_mm"] * h
        vol = vol_base + vol_fins

    else:
        msg = f"Unknown shape '{shape}'"
        raise ValueError(msg)

    # add caps
    r_cap = p.get("body_outer_r_mm", r_in + 3.0)
    vol += 2 * math.pi * r_cap**2 * p["cap_t_mm"]

    # convert mm³ → cm³ → g
    return vol * 1e-3 * density_g_cm3


def compute_pen_sv_ratio(shape: str, detector_type: str) -> float:
    """Compute surface-to-volume ratio of PEN material in 1/mm.

    Includes all PEN surfaces in contact with LAr:
    inner surface, outer surface, top and bottom edges.
    """
    p = PEN_ENCLOSURES[detector_type]
    r_in = p["body_inner_r_mm"]
    h = p["body_h_mm"]

    if shape == "flat":
        r_out = p["body_outer_r_mm"]
        surf = (
            2 * math.pi * r_in * h  # inner
            + 2 * math.pi * r_out * h  # outer
            + 2 * math.pi * (r_out**2 - r_in**2)
        )  # top + bottom edges
        vol = math.pi * (r_out**2 - r_in**2) * h

    elif shape == "ribbed":
        base_wall = 0.5
        n_ribs = max(1, int(h / p["rib_spacing_mm"]))
        r_base = r_in + base_wall
        r_rib = r_base + p["rib_t_mm"]

        # base shell surfaces
        surf_base = (
            2 * math.pi * r_in * h  # inner
            + 2 * math.pi * r_base * h  # outer base
            + 2 * math.pi * (r_base**2 - r_in**2)
        )  # edges

        # rib surfaces — top, bottom, outer
        surf_ribs = n_ribs * (
            2 * math.pi * (r_rib**2 - r_base**2)  # top face
            + 2 * math.pi * (r_rib**2 - r_base**2)  # bottom face
            + 2 * math.pi * r_rib * p["rib_h_mm"]  # outer curved
        )

        surf = surf_base + surf_ribs
        vol = math.pi * (r_base**2 - r_in**2) * h + n_ribs * math.pi * (r_rib**2 - r_base**2) * p["rib_h_mm"]

    elif shape == "slotted":
        r_out = p["body_outer_r_mm"]
        wall_t = r_out - r_in
        n_slots = p["n_slots"]
        slot_w = p["slot_w_mm"]
        slot_h = p["slot_h_mm"]

        # shell surfaces minus slot openings plus slot walls
        surf_shell = 2 * math.pi * r_in * h + 2 * math.pi * r_out * h + 2 * math.pi * (r_out**2 - r_in**2)
        # slot walls (2 long sides + 2 short sides per slot)
        surf_slots = n_slots * (2 * slot_h * wall_t + 2 * slot_w * wall_t)
        # subtract slot openings from inner/outer surfaces
        surf_openings = n_slots * slot_w * slot_h * 2
        surf = surf_shell + surf_slots - surf_openings

        vol_shell = math.pi * (r_out**2 - r_in**2) * h
        vol_slots = n_slots * slot_w * wall_t * slot_h
        vol = vol_shell - vol_slots

    elif shape == "layered":
        surf = 0.0
        vol = 0.0
        for i in range(p["n_shells"]):
            r_s_in = r_in + i * (p["shell_t_mm"] + p["gap_t_mm"])
            r_s_out = r_s_in + p["shell_t_mm"]
            surf += (
                2 * math.pi * r_s_in * h + 2 * math.pi * r_s_out * h + 2 * math.pi * (r_s_out**2 - r_s_in**2)
            )
            vol += math.pi * (r_s_out**2 - r_s_in**2) * h

    elif shape == "honeycomb":
        r_out = p["body_outer_r_mm"]
        n_cells = max(6, int(TWO_PI * r_in / p["cell_size_mm"]))
        wall_t = p["shell_t_mm"] / r_in
        fill = 1.0 - wall_t / (TWO_PI / n_cells)

        # curved inner/outer surfaces scaled by fill fraction
        surf_curved = 2 * math.pi * r_in * h * fill + 2 * math.pi * r_out * h * fill
        # flat azimuthal wall faces (2 per gap per ring)
        n_rings = max(2, int(h / p["cell_size_mm"]))
        ring_h = h / n_rings
        surf_walls = n_rings * n_cells * 2 * (r_out - r_in) * ring_h
        # top/bottom edges
        surf_edges = 2 * math.pi * (r_out**2 - r_in**2) * fill

        surf = surf_curved + surf_walls + surf_edges
        vol = math.pi * (r_out**2 - r_in**2) * h * fill

    elif shape == "fins":
        base_wall = 0.5
        r_base = r_in + base_wall
        n_fins = p["n_fins"]
        fin_l = p["fin_l_mm"]
        fin_t = p["fin_t_mm"]

        # base shell surfaces
        surf_base = 2 * math.pi * r_in * h + 2 * math.pi * r_base * h + 2 * math.pi * (r_base**2 - r_in**2)
        # fin surfaces — 2 flat sides + tip + top/bottom edges per fin
        surf_fins = n_fins * (
            2 * fin_l * h  # two flat sides
            + fin_t * h  # tip
            + 2 * fin_l * fin_t
        )  # top/bottom

        surf = surf_base + surf_fins
        vol = math.pi * (r_base**2 - r_in**2) * h + n_fins * fin_t * fin_l * h

    else:
        msg = f"Unknown shape '{shape}'"
        raise ValueError(msg)

    return surf / vol if vol > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Shape registry
# ─────────────────────────────────────────────────────────────────────────────

PEN_SHAPE_BUILDERS = {
    "flat": build_pen_polycone,
    "ribbed": build_pen_ribbed_shell,
    "slotted": build_pen_slotted_shell,
    "layered": build_pen_layered_shells,
    "honeycomb": build_pen_honeycomb,
    "fins": build_pen_fins,
}


def build_pen_shape(
    shape: str,
    name: str,
    detector_type: str,
    registry: Registry,
) -> solid.Union:
    """Build a PEN enclosure of the given shape for a detector type.

    Parameters
    ----------
    shape
        One of 'flat', 'ribbed', 'slotted', 'layered', 'honeycomb', 'fins'.
    name
        Base name for the solid.
    detector_type
        One of 'bege', 'icpc'.
    registry
        pyg4ometry registry.
    """
    if shape not in PEN_SHAPE_BUILDERS:
        msg = f"Unknown shape '{shape}'. Available: {list(PEN_SHAPE_BUILDERS)}"
        raise KeyError(msg)
    if detector_type not in PEN_ENCLOSURES:
        msg = f"Unknown detector type '{detector_type}'. Available: {list(PEN_ENCLOSURES)}"
        raise KeyError(msg)

    return PEN_SHAPE_BUILDERS[shape](name, registry=registry, **PEN_ENCLOSURES[detector_type])


def build_pen_enclosure(detector_type: str, registry: Registry) -> solid.Union:
    """Build the default (flat) PEN enclosure for a given detector type.

    Parameters
    ----------
    detector_type
        One of 'bege', 'icpc'.
    registry
        pyg4ometry registry.
    """
    if detector_type not in PEN_ENCLOSURES:
        msg = f"Unknown detector type '{detector_type}'. Available: {list(PEN_ENCLOSURES)}"
        raise KeyError(msg)

    return build_pen_polycone(
        detector_type,
        registry=registry,
        **PEN_ENCLOSURES[detector_type],
    )
