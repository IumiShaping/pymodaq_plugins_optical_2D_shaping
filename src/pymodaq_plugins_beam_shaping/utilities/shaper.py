import dataclasses

from pymodaq_utils.enums import StrEnum
from pymodaq_utils.config import GlobalConfig

config = GlobalConfig()


class ModulatorType(StrEnum):
    PHASE = 'phase'
    AMPLITUDE = 'amplitude'
    BOTH = 'both'


@dataclasses.dataclass
class Shaper:
    name: str
    width: int
    height: int
    pixel_size: float  # in micrometer
    has_internal_calibration: bool
    use_internal_calibration: bool
    modulator_type: ModulatorType


def get_shaper() -> Shaper:
    default_slm = config('beam_shaping', 'SLM', 'default_slm')[0]
    return Shaper(name=default_slm,
                  width=config('beam_shaping', 'SLM', default_slm, 'width'),
                  height=config('beam_shaping', 'SLM', default_slm, 'height'),
                  pixel_size=config('beam_shaping', 'SLM', default_slm, 'pixel_size'),
                  has_internal_calibration=config('beam_shaping', 'SLM', default_slm, 'has_internal_calibration'),
                  use_internal_calibration=config('beam_shaping', 'SLM', default_slm, 'use_internal_calibration'),
                  modulator_type=ModulatorType[config('beam_shaping', 'SLM', default_slm, 'modulator_type')[0]]
                  )