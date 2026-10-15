"""WISE L1b frame mask bits (Explanatory Supplement, All-Sky release,
section IV.4.a.i.2, Table 2; NEOWISE-R frames use the same definitions).

Bits 0-7 are static (bad hardware); the rest are set per frame. Bit 27 is
deliberately *not* treated as bad: it marks temporal outliers found when the
co-add pipeline stacks overlapping frames -- which is exactly what a moving
asteroid is.
"""

BITS = {
    0: "static: excessively noisy (high dark current)",
    1: "static: generally noisy",
    2: "static: dead or very low responsivity",
    3: "static: low responsivity or low dark current",
    4: "static: high responsivity or high dark current",
    5: "static: saturated anywhere in ramp",
    6: "static: high, uncertain or unreliable non-linearity",
    7: "static: broken hardware pixel or noisy responsivity",
    9: "broken pixel or negative slope fit",
    **{b: f"saturated in sample read {b - 9}" for b in range(10, 19)},
    21: "new/transient bad pixel (dynamic masking)",
    26: "non-linearity correction unreliable",
    27: "temporal outlier in co-add stacking (moving objects too!)",
    28: "spike outlier (single-frame glitch filter)",
}

# Bits that make a pixel unusable for the moving-object search.
BAD_BITS_LIST = [0, 1, 2, 3, 4, 6, 7, 9, *range(10, 19), 21, 26, 28]
BAD_BITS = sum(1 << b for b in BAD_BITS_LIST)


def describe(value: int) -> list:
    """Names of the bits set in a mask value."""
    return [BITS.get(b, f"bit {b} (reserved)") for b in range(31) if value >> b & 1]
