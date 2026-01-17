from basketworld.envs.core import geometry


def test_offset_axial_roundtrip():
    for row in range(4):
        for col in range(4):
            q, r = geometry.offset_to_axial_formula(col, row)
            col2, row2 = geometry.axial_to_offset_formula(q, r)
            assert (col, row) == (col2, row2)


def test_precompute_coord_caches_and_distance():
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    offset_cache, axial_to_offset, axial_to_cart, valid = geometry.precompute_coord_caches(
        court_width=2, court_height=2, cells=cells
    )
    # Caches should include all cells
    assert len(offset_cache) == 2 and len(offset_cache[0]) == 2
    assert (0, 0) in valid and (1, 1) in valid
    assert axial_to_offset[(0, 0)] == (0, 0)
    assert axial_to_offset[(1, 1)] == (1, 1)
    # Cartesian cache should match formula
    assert axial_to_cart[(1, 0)] == geometry.axial_to_cartesian_formula(1, 0)

    lut, index = geometry.precompute_hex_distance_lut(cells)
    d_formula = geometry.hex_distance_formula(0, 0, 1, 1)
    assert lut[index[(0, 0)], index[(1, 1)]] == d_formula
