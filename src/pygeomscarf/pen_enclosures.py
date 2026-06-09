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

Geometry strategy
-----------------
The fixed mechanical specs (wall thickness, flange geometry, cap
thickness, shape tuning parameters) are stored in PEN_ENCLOSURES and
come from the SCARF PEN mechanical drawings — they are the same for
all detectors of a given type.

The detector-dependent dimensions (crystal radius, crystal height,
borehole radius and depth) are read from the detector metadata at
runtime via pen_params_from_meta(). This ensures the PEN enclosure
is always correctly sized around whatever crystal it wraps, with no
hardcoded crystal dimensions that could cause overlaps.

    PEN_ENCLOSURES[det_type]   → fixed hardware: wall, cap, flange, shape params
    hpge_meta["geometry"]      → variable: det_r, det_h, borehole dims
            ↓
    pen_params_from_meta()     → merged + validated final params
            ↓
    build_pen_shape()          → correctly sized solid, overlap-free

Clearances
----------
GAP_MM = 0.5   radial air gap between crystal outer surface and PEN inner wall
WALL_MM = 1.5  PEN barrel wall thickness (from drawing)
CAP_MM = 1.5   PEN cap thickness (from drawing)
COPLANAR_SEPARATION_MM = 0.01  avoids coplanar surface artefacts in boolean ops

Validation performed in pen_params_from_meta
--------------------------------------------
1. flange_inner_r_mm  clamped to >= body_inner_r_mm   (prevents flange/crystal overlap)
2. slot_h_mm          clamped to <= body_h_mm - 2mm    (slots can't exceed barrel height)
3. fin r_outer        = body_inner_r_mm + base_wall + fin_l_mm  (caps cover fin tips)
4. n_layers           included in PEN_ENCLOSURES and passed through to honeycomb builder

Borehole plug (ICPC only)
--------------------------
The plug fills the HPGe borehole. In pygeomhpges convention the borehole
is at the TOP of the crystal (+z face), so the plug centre sits at:
    plug_z = +det_h/2 - borehole_half_height - COPLANAR_SEPARATION_MM
Its radius and depth are read directly from:
    geometry.borehole.radius_in_mm
    geometry.borehole.depth_in_mm
Both are reduced by COPLANAR_SEPARATION_MM to avoid surface coincidence
with the crystal borehole wall and face.

Shapes available
----------------
flat        — baseline cylindrical shell (current design)
ribbed      — thin base shell + horizontal ribs
slotted     — shell with LAr-filled rectangular slots
layered     — multiple concentric shells with LAr gaps
honeycomb   — azimuthal segments alternating with LAr gaps
fins        — thin base shell + outward-pointing radial fins

All shapes are conformal — the inner surface hugs the HPGe detector
with minimal clearance to maximise particle interception.

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

# Radial air gap between crystal surface and PEN inner wall
GAP_MM = 0.5
# PEN barrel wall thickness (from SCARF mechanical drawing)
WALL_MM = 1.5
# Thin base shell thickness used by ribbed/fins shapes
BASE_WALL_MM = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Fixed hardware specs — from SCARF PEN mechanical drawings
# Crystal dimensions are NOT stored here; they come from metadata at runtime.
# ─────────────────────────────────────────────────────────────────────────────

PEN_ENCLOSURES = {
    "bege": {
        # PEN hardware dims from drawing (fixed regardless of which crystal)
        "cap_t_mm": 1.5,
        "flange_t_mm": 1.5,
        "flange_outer_r_mm": 44.5,
        "flange_inner_r_mm": 33.75,  # validated at runtime: clamped >= body_inner_r_mm
        # shape tuning parameters
        "n_shells": 3,
        "shell_t_mm": 2.0,
        "gap_t_mm": 3.0,
        "n_ribs": 6,
        "rib_t_mm": 3.0,
        "rib_h_mm": 2.0,
        "rib_spacing_mm": 5.0,
        "n_slots": 8,
        "slot_w_mm": 8.0,
        "slot_h_mm": 20.0,  # validated at runtime: clamped <= body_h_mm - 2mm
        "cell_size_mm": 1.5,
        "n_layers": 1,  # honeycomb radial layers
        "n_fins": 16,
        "fin_l_mm": 4.0,
        "fin_t_mm": 1.0,
    },
    "icpc": {
        # PEN hardware dims from drawing (fixed regardless of which crystal)
        "cap_t_mm": 1.5,
        "flange_t_mm": 1.5,
        "flange_outer_r_mm": 46.0,
        "flange_inner_r_mm": 28.0,  # validated at runtime: clamped >= body_inner_r_mm
        # shape tuning parameters
        "n_shells": 2,
        "shell_t_mm": 1.5,
        "gap_t_mm": 2.0,
        "n_ribs": 8,
        "rib_t_mm": 3.0,
        "rib_h_mm": 2.0,
        "rib_spacing_mm": 8.0,
        "n_slots": 8,
        "slot_w_mm": 8.0,
        "slot_h_mm": 40.0,  # validated at runtime: clamped <= body_h_mm - 2mm
        "cell_size_mm": 1.5,
        "n_layers": 1,  # honeycomb radial layers
        "n_fins": 20,
        "fin_l_mm": 3.0,
        "fin_t_mm": 1.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata → geometry parameter derivation + validation
# ─────────────────────────────────────────────────────────────────────────────


def pen_params_from_meta(det_meta: dict, detector_type: str) -> dict:
    """Derive and validate full PEN enclosure params from detector metadata.

    Merges fixed hardware specs from PEN_ENCLOSURES with crystal geometry
    read from det_meta["geometry"]. Performs all cross-parameter validation
    so individual shape builders receive only safe, consistent values.

    Validations applied
    -------------------
    - flange_inner_r_mm  clamped to >= body_inner_r_mm
      (flange bore must not be smaller than barrel bore or it clips the crystal)
    - slot_h_mm          clamped to body_h_mm - 2.0
      (slots cannot exceed barrel height; 1mm margin top and bottom)
    - fin r_outer        set to body_inner_r_mm + BASE_WALL_MM + fin_l_mm
      (caps must cover the full radial extent of the fins)

    Parameters
    ----------
    det_meta
        Full detector metadata dict (loaded yaml), e.g. det_meta[name]
        from strings.py. Must contain a 'geometry' block.
    detector_type
        One of 'bege', 'icpc'. Selects the hardware spec from PEN_ENCLOSURES.

    Returns
    -------
    dict
        Complete, validated parameter dict ready to unpack into any shape builder.
    """
    params = dict(PEN_ENCLOSURES[detector_type])

    geo = det_meta["geometry"]
    det_r = geo["radius_in_mm"]
    det_h = geo["height_in_mm"]

    # --- crystal-derived dimensions ---
    params["det_r_mm"] = det_r
    params["det_h_mm"] = det_h
    params["body_inner_r_mm"] = det_r + GAP_MM  # 0.5mm radial clearance
    params["body_outer_r_mm"] = det_r + GAP_MM + WALL_MM  # + 1.5mm wall
    params["body_h_mm"] = det_h  # barrel spans full crystal height
    params["z_offset_mm"] = det_h / 2.0  # placement offset in LAr frame

    # --- borehole plug: ICPC only, always from metadata ---
    # V07302A: radius=4mm, depth=35mm
    # icpc_test: radius=5mm, depth=32mm
    # Both shrunk by COPLANAR_SEPARATION_MM to avoid surface coincidence
    if "borehole" in geo:
        params["borehole_r_mm"] = geo["borehole"]["radius_in_mm"] - COPLANAR_SEPARATION_MM
        params["borehole_h_mm"] = geo["borehole"]["depth_in_mm"] - COPLANAR_SEPARATION_MM
    else:
        params["borehole_r_mm"] = None
        params["borehole_h_mm"] = None

    # --- validation 1: flange inner bore must not clip the crystal ---
    # flange_inner_r_mm from drawing may be smaller than body_inner_r_mm
    # for a larger crystal, which would cause the flange to overlap the crystal
    # at the top/bottom edges. Clamp to be safe.
    body_inner = params["body_inner_r_mm"]
    params["flange_inner_r_mm"] = max(params["flange_inner_r_mm"], body_inner)

    # --- validation 2: slot height must not exceed barrel height ---
    # slot_h_mm is tuned per detector type but body_h_mm now comes from
    # metadata and may differ. A 1mm margin at top and bottom is preserved.
    max_slot_h = det_h - 2.0
    params["slot_h_mm"] = min(params["slot_h_mm"], max_slot_h)

    # --- validation 3: fin r_outer must cover fin tips for cap sizing ---
    # fins extend radially outward from body_inner_r_mm + BASE_WALL_MM
    # by fin_l_mm. The cap outer radius must reach the fin tips.
    params["fin_r_outer_mm"] = body_inner + BASE_WALL_MM + params["fin_l_mm"]

    return params


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
    """Add a PEN plug into the HPGe borehole.

    In pygeomhpges convention the borehole is at the TOP of the crystal
    (+z face). The plug centre sits at:
        +det_h/2 - borehole_half_height - COPLANAR_SEPARATION_MM

    Both radius and depth are already reduced by COPLANAR_SEPARATION_MM
    before being passed in (done in pen_params_from_meta).
    """
    borehole_half_height = borehole_h_mm / 2.0

    plug = solid.Tubs(
        f"{name}_borehole_plug",
        0,
        borehole_r_mm,
        borehole_half_height,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    if det_h_mm > 0:
        # borehole is at the TOP of the crystal in pygeomhpges convention (+z)
        # plug centre sits just below the top face, extending downward
        plug_z = +det_h_mm / 2.0 - borehole_half_height - COPLANAR_SEPARATION_MM
    else:
        plug_z = +body_half_height - borehole_half_height - COPLANAR_SEPARATION_MM

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
        f"{name}_cap_top",
        r_inner,
        r_outer,
        cap_half,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )
    cap_bot = solid.Tubs(
        f"{name}_cap_bot",
        r_inner,
        r_outer,
        cap_half,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
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
    det_h_mm: float = 0.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Baseline flat conformal cylindrical shell hugging HPGe surface."""
    body_half_height = body_h_mm / 2.0

    body = solid.Tubs(
        f"{name}_body",
        body_inner_r_mm,
        body_outer_r_mm,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    shell = _add_caps_and_flanges(
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

    if borehole_r_mm is not None and borehole_h_mm is not None:
        shell = _add_borehole_plug(
            name,
            shell,
            body_half_height,
            flange_t_mm,
            cap_t_mm,
            borehole_r_mm,
            borehole_h_mm,
            registry,
            det_h_mm=det_h_mm,
        )

    return shell


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
    det_h_mm: float = 0.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal thin base shell with horizontal annular ribs.

    n_ribs is computed from body_h_mm / rib_spacing_mm so it scales
    automatically with crystal height from metadata.
    """
    body_half_height = body_h_mm / 2.0

    base = solid.Tubs(
        f"{name}_base",
        body_inner_r_mm,
        body_inner_r_mm + BASE_WALL_MM,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    # n_ribs scales with crystal height — no hardcoding needed
    n_ribs = max(1, int(body_h_mm / rib_spacing_mm))
    shell = base
    for i in range(n_ribs):
        z_pos = -body_half_height + (i + 0.5) * rib_spacing_mm
        rib = solid.Tubs(
            f"{name}_rib_{i}",
            body_inner_r_mm,
            body_inner_r_mm + BASE_WALL_MM + rib_t_mm,
            rib_h_mm / 2.0,
            0,
            TWO_PI,
            registry=registry,
            lunit="mm",
        )
        shell = solid.Union(
            f"{name}_u_rib_{i}",
            shell,
            rib,
            [[0, 0, 0], [0, 0, z_pos]],
            registry,
        )

    r_outer = body_inner_r_mm + BASE_WALL_MM + rib_t_mm
    shell = _add_caps_and_flanges(
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

    if borehole_r_mm is not None and borehole_h_mm is not None:
        shell = _add_borehole_plug(
            name,
            shell,
            body_half_height,
            flange_t_mm,
            cap_t_mm,
            borehole_r_mm,
            borehole_h_mm,
            registry,
            det_h_mm=det_h_mm,
        )

    return shell


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
    det_h_mm: float = 0.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal shell with rectangular LAr-filled slots.

    slot_h_mm is already clamped to body_h_mm - 2mm by pen_params_from_meta,
    so slots always fit within the barrel regardless of crystal height.
    """
    body_half_height = body_h_mm / 2.0
    wall_t = body_outer_r_mm - body_inner_r_mm

    base = solid.Tubs(
        f"{name}_base",
        body_inner_r_mm,
        body_outer_r_mm,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    shell = base
    for i in range(n_slots):
        angle = i * TWO_PI / n_slots
        slot = solid.Box(
            f"{name}_slot_{i}",
            slot_w_mm,
            wall_t + 2.0,
            slot_h_mm,
            registry=registry,
            lunit="mm",
        )
        x = (body_inner_r_mm + wall_t / 2.0) * math.cos(angle)
        y = (body_inner_r_mm + wall_t / 2.0) * math.sin(angle)
        shell = solid.Subtraction(
            f"{name}_slot_cut_{i}",
            shell,
            slot,
            [[0, 0, angle], [x, y, 0]],
            registry,
        )

    shell = _add_caps_and_flanges(
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

    if borehole_r_mm is not None and borehole_h_mm is not None:
        shell = _add_borehole_plug(
            name,
            shell,
            body_half_height,
            flange_t_mm,
            cap_t_mm,
            borehole_r_mm,
            borehole_h_mm,
            registry,
            det_h_mm=det_h_mm,
        )

    return shell


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
    det_h_mm: float = 0.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal concentric multi-shell PEN enclosure with LAr gaps."""
    if n_shells < 1:
        msg = f"n_shells must be >= 1, got {n_shells}"
        raise ValueError(msg)

    body_half_height = body_h_mm / 2.0
    shell = None

    for i in range(n_shells):
        r_in = body_inner_r_mm + i * (shell_t_mm + gap_t_mm)
        r_out = r_in + shell_t_mm
        tub = solid.Tubs(
            f"{name}_shell_{i}",
            r_in,
            r_out,
            body_h_mm,
            0,
            TWO_PI,
            registry=registry,
            lunit="mm",
        )
        shell = (
            tub
            if shell is None
            else solid.Union(
                f"{name}_u_shell_{i}",
                shell,
                tub,
                [[0, 0, 0], [0, 0, 0]],
                registry,
            )
        )

    r_outer = body_inner_r_mm + n_shells * shell_t_mm + (n_shells - 1) * gap_t_mm

    shell = _add_caps_and_flanges(
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

    if borehole_r_mm is not None and borehole_h_mm is not None:
        shell = _add_borehole_plug(
            name,
            shell,
            body_half_height,
            flange_t_mm,
            cap_t_mm,
            borehole_r_mm,
            borehole_h_mm,
            registry,
            det_h_mm=det_h_mm,
        )

    return shell


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
    det_h_mm: float = 0.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal honeycomb PEN enclosure with continuous hexagonal lattice.

    n_layers is now read from PEN_ENCLOSURES (previously defaulted to 2
    and was never passed through). r_outer is computed from n_layers so
    it scales correctly with the chosen honeycomb depth.
    """
    body_half_height = body_h_mm / 2.0
    cell_r = cell_size_mm
    wall_t = shell_t_mm
    r_inner = body_inner_r_mm
    r_outer = r_inner + n_layers * (2 * cell_r + wall_t)

    dx = (2 * cell_r + wall_t) * math.cos(math.radians(30))
    dy = 2 * cell_r + wall_t

    def _hex_verts(r):
        return [
            [r * math.cos(math.radians(30 + 60 * i)), r * math.sin(math.radians(30 + 60 * i))]
            for i in range(6)
        ]

    centers = []
    rows = int((r_outer + cell_r) / dy) + 3
    cols = int((r_outer + cell_r) / dx) + 3
    for row in range(-rows, rows + 1):
        for col in range(-cols, cols + 1):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy * math.sqrt(3) / 2
            d = math.sqrt(x**2 + y**2)
            if r_inner < d < r_outer:
                centers.append((x, y))

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

    shell = _add_caps_and_flanges(
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

    if borehole_r_mm is not None and borehole_h_mm is not None:
        shell = _add_borehole_plug(
            name,
            shell,
            body_half_height,
            flange_t_mm,
            cap_t_mm,
            borehole_r_mm,
            borehole_h_mm,
            registry,
            det_h_mm=det_h_mm,
        )

    return shell


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
    fin_r_outer_mm: float = 0.0,
    det_h_mm: float = 0.0,
    borehole_r_mm: float | None = None,
    borehole_h_mm: float | None = None,
    registry: Registry,
    **_: object,
) -> solid.Union:
    """Conformal thin base shell with outward-pointing radial fins.

    fin_r_outer_mm is set by pen_params_from_meta to
    body_inner_r_mm + BASE_WALL_MM + fin_l_mm, ensuring the cap outer
    radius always covers the full radial extent of the fins.
    """
    body_half_height = body_h_mm / 2.0

    base = solid.Tubs(
        f"{name}_base",
        body_inner_r_mm,
        body_inner_r_mm + BASE_WALL_MM,
        body_h_mm,
        0,
        TWO_PI,
        registry=registry,
        lunit="mm",
    )

    shell = base
    for i in range(n_fins):
        angle = i * TWO_PI / n_fins
        fin_center_r = body_inner_r_mm + BASE_WALL_MM + fin_l_mm / 2.0
        fx = fin_center_r * math.cos(angle)
        fy = fin_center_r * math.sin(angle)

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

    # use fin_r_outer_mm (= base + BASE_WALL_MM + fin_l) so caps cover fin tips
    r_outer = fin_r_outer_mm if fin_r_outer_mm > 0 else body_inner_r_mm + BASE_WALL_MM + fin_l_mm

    shell = _add_caps_and_flanges(
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

    if borehole_r_mm is not None and borehole_h_mm is not None:
        shell = _add_borehole_plug(
            name,
            shell,
            body_half_height,
            flange_t_mm,
            cap_t_mm,
            borehole_r_mm,
            borehole_h_mm,
            registry,
            det_h_mm=det_h_mm,
        )

    return shell


# ─────────────────────────────────────────────────────────────────────────────
# Mass and S/V calculators
# ─────────────────────────────────────────────────────────────────────────────


def compute_pen_mass_g(
    shape: str,
    detector_type: str,
    det_meta: dict,
    density_g_cm3: float = PEN_DENSITY_G_CM3,
) -> float:
    """Compute approximate total PEN mass in grams for a given shape.

    Uses the same validated params as the geometry builders so mass
    estimates are consistent with the actual solid built.
    """
    p = pen_params_from_meta(det_meta, detector_type)
    r_in = p["body_inner_r_mm"]
    h = p["body_h_mm"]

    if shape == "flat":
        r_out = p["body_outer_r_mm"]
        vol = math.pi * (r_out**2 - r_in**2) * h

    elif shape == "ribbed":
        n_ribs = max(1, int(h / p["rib_spacing_mm"]))
        r_base = r_in + BASE_WALL_MM
        r_rib = r_base + p["rib_t_mm"]
        vol_base = math.pi * (r_base**2 - r_in**2) * h
        vol_ribs = n_ribs * math.pi * (r_rib**2 - r_base**2) * p["rib_h_mm"]
        vol = vol_base + vol_ribs

    elif shape == "slotted":
        r_out = p["body_outer_r_mm"]
        wall_t = r_out - r_in
        vol_shell = math.pi * (r_out**2 - r_in**2) * h
        vol_slots = p["n_slots"] * p["slot_w_mm"] * wall_t * p["slot_h_mm"]
        vol = vol_shell - vol_slots

    elif shape == "layered":
        vol = 0.0
        for i in range(p["n_shells"]):
            r_s_in = r_in + i * (p["shell_t_mm"] + p["gap_t_mm"])
            r_s_out = r_s_in + p["shell_t_mm"]
            vol += math.pi * (r_s_out**2 - r_s_in**2) * h

    elif shape == "honeycomb":
        n_layers = p["n_layers"]
        cell_r = p["cell_size_mm"]
        wall_t = p["shell_t_mm"]
        r_out = r_in + n_layers * (2 * cell_r + wall_t)
        n_cells = max(6, int(TWO_PI * r_in / cell_r))
        fill_fraction = 1.0 - (wall_t / r_in) / (TWO_PI / n_cells)
        vol = math.pi * (r_out**2 - r_in**2) * h * fill_fraction

    elif shape == "fins":
        r_base = r_in + BASE_WALL_MM
        vol_base = math.pi * (r_base**2 - r_in**2) * h
        vol_fins = p["n_fins"] * p["fin_t_mm"] * p["fin_l_mm"] * h
        vol = vol_base + vol_fins

    else:
        msg = f"Unknown shape '{shape}'"
        raise ValueError(msg)

    # add caps (use body_outer_r_mm as cap radius)
    r_cap = p["body_outer_r_mm"]
    vol += 2 * math.pi * r_cap**2 * p["cap_t_mm"]

    return vol * 1e-3 * density_g_cm3


def compute_pen_sv_ratio(shape: str, detector_type: str, det_meta: dict) -> float:
    """Compute surface-to-volume ratio of PEN material in 1/mm."""
    p = pen_params_from_meta(det_meta, detector_type)
    r_in = p["body_inner_r_mm"]
    h = p["body_h_mm"]

    if shape == "flat":
        r_out = p["body_outer_r_mm"]
        surf = 2 * math.pi * r_in * h + 2 * math.pi * r_out * h + 2 * math.pi * (r_out**2 - r_in**2)
        vol = math.pi * (r_out**2 - r_in**2) * h

    elif shape == "ribbed":
        n_ribs = max(1, int(h / p["rib_spacing_mm"]))
        r_base = r_in + BASE_WALL_MM
        r_rib = r_base + p["rib_t_mm"]
        surf = (
            2 * math.pi * r_in * h
            + 2 * math.pi * r_base * h
            + 2 * math.pi * (r_base**2 - r_in**2)
            + n_ribs * (4 * math.pi * (r_rib**2 - r_base**2) + 2 * math.pi * r_rib * p["rib_h_mm"])
        )
        vol = math.pi * (r_base**2 - r_in**2) * h + n_ribs * math.pi * (r_rib**2 - r_base**2) * p["rib_h_mm"]

    elif shape == "slotted":
        r_out = p["body_outer_r_mm"]
        wall_t = r_out - r_in
        n_s, sw, sh = p["n_slots"], p["slot_w_mm"], p["slot_h_mm"]
        surf = (
            2 * math.pi * r_in * h
            + 2 * math.pi * r_out * h
            + 2 * math.pi * (r_out**2 - r_in**2)
            + n_s * (2 * sh * wall_t + 2 * sw * wall_t)
            - n_s * sw * sh * 2
        )
        vol = math.pi * (r_out**2 - r_in**2) * h - n_s * sw * wall_t * sh

    elif shape == "layered":
        surf = vol = 0.0
        for i in range(p["n_shells"]):
            r_s_in = r_in + i * (p["shell_t_mm"] + p["gap_t_mm"])
            r_s_out = r_s_in + p["shell_t_mm"]
            surf += (
                2 * math.pi * r_s_in * h + 2 * math.pi * r_s_out * h + 2 * math.pi * (r_s_out**2 - r_s_in**2)
            )
            vol += math.pi * (r_s_out**2 - r_s_in**2) * h

    elif shape == "honeycomb":
        n_layers = p["n_layers"]
        cell_r = p["cell_size_mm"]
        wall_t = p["shell_t_mm"]
        r_out = r_in + n_layers * (2 * cell_r + wall_t)
        n_cells = max(6, int(TWO_PI * r_in / cell_r))
        fill = 1.0 - (wall_t / r_in) / (TWO_PI / n_cells)
        n_rings = max(2, int(h / cell_r))
        ring_h = h / n_rings
        surf = (
            2 * math.pi * r_in * h * fill
            + 2 * math.pi * r_out * h * fill
            + n_rings * n_cells * 2 * (r_out - r_in) * ring_h
            + 2 * math.pi * (r_out**2 - r_in**2) * fill
        )
        vol = math.pi * (r_out**2 - r_in**2) * h * fill

    elif shape == "fins":
        r_base = r_in + BASE_WALL_MM
        n_fins = p["n_fins"]
        fin_l = p["fin_l_mm"]
        fin_t = p["fin_t_mm"]
        surf = (
            2 * math.pi * r_in * h
            + 2 * math.pi * r_base * h
            + 2 * math.pi * (r_base**2 - r_in**2)
            + n_fins * (2 * fin_l * h + fin_t * h + 2 * fin_l * fin_t)
        )
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
    det_meta: dict,
) -> solid.Union:
    """Build a PEN enclosure of the given shape for a specific detector.

    All geometry-dependent dimensions (crystal radius, height, borehole)
    are read from det_meta["geometry"] and validated before being passed
    to the shape builder. Fixed hardware specs (wall, cap, flange, shape
    tuning) come from PEN_ENCLOSURES[detector_type].

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
    det_meta
        Full detector metadata dict (det_meta[name] from strings.py).
        Must contain a 'geometry' block with at minimum:
            radius_in_mm, height_in_mm
        And for ICPC:
            borehole.radius_in_mm, borehole.depth_in_mm
    """
    if shape not in PEN_SHAPE_BUILDERS:
        msg = f"Unknown shape '{shape}'. Available: {list(PEN_SHAPE_BUILDERS)}"
        raise KeyError(msg)
    if detector_type not in PEN_ENCLOSURES:
        msg = f"Unknown detector type '{detector_type}'. Available: {list(PEN_ENCLOSURES)}"
        raise KeyError(msg)

    params = pen_params_from_meta(det_meta, detector_type)
    return PEN_SHAPE_BUILDERS[shape](name, registry=registry, **params)


def build_pen_enclosure(detector_type: str, registry: Registry, det_meta: dict) -> solid.Union:
    """Build the default (flat) PEN enclosure for a given detector."""
    return build_pen_shape("flat", detector_type, detector_type, registry, det_meta)
