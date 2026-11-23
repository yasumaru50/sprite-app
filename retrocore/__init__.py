from .dither import DITHER_METHODS, apply_palette
from .palette import (
    EXTRACT_METHODS,
    PRESET_PALETTES,
    RampPalette,
    extract_palette,
    extract_ramp_palette,
    load_palette,
    save_palette,
    snap_to_pc98,
)
from .pipeline import (
    CONVERT_MODES,
    FIT_MODES,
    PALETTE_SOURCES,
    RESAMPLE_MODES,
    RESOLUTION_PRESETS,
    ConvertResult,
    ConvertSettings,
    convert,
    load_image,
    save_image,
)
