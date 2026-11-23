from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CheckedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Mapping(CheckedModel):
    energy: int = Field(default=0, ge=0, le=63)
    signal: int = Field(default=1, ge=0, le=63)
    reference: int = Field(default=2, ge=0, le=63)
    mode: Literal["mu", "transmission", "fluorescence"] = "mu"
    energy_unit: Literal["eV", "keV"] = "eV"


class Parameters(CheckedModel):
    e0: float | None = Field(default=None, gt=0, lt=200000)
    energy_shift: float = Field(default=0, ge=-1000, le=1000)
    pre1: float = Field(default=-150, ge=-10000, lt=0)
    pre2: float = Field(default=-30, ge=-10000, lt=0)
    norm1: float = Field(default=150, gt=0, le=20000)
    norm2: float = Field(default=700, gt=0, le=20000)
    nnorm: int = Field(default=2, ge=0, le=3)
    rbkg: float = Field(default=1, ge=0.3, le=2.5)
    bkg_kweight: int = Field(default=1, ge=0, le=3)
    bkg_kmax: float | None = Field(default=None, gt=0, le=100)
    kmin: float = Field(default=2, ge=0, le=30)
    kmax: float = Field(default=12, gt=0, le=30)
    kweight: int = Field(default=2, ge=0, le=3)
    dk: float = Field(default=1, ge=0.1, le=5)

    @model_validator(mode="after")
    def ordered_windows(self):
        if self.pre1 >= self.pre2 or self.norm1 >= self.norm2:
            raise ValueError("Each fitting window must have its lower limit below its upper limit.")
        if self.kmax - self.kmin <= self.dk:
            raise ValueError("FT kmax − kmin must exceed the taper width dk.")
        if self.kmin < self.dk / 2:
            raise ValueError("FT kmin must be at least dk/2 so the full taper is supported.")
        if self.bkg_kmax is not None and self.kmax + self.dk / 2 > self.bkg_kmax:
            raise ValueError("The FT taper must stay within the declared background-processing range.")
        return self


class AnalysisRequest(CheckedModel):
    processing_profile: Literal['manual', 'adaptive'] = 'manual'
    object_id: str | None = Field(default=None, max_length=80)
    assessment_token: str | None = Field(default=None, max_length=16000)
    filename: str = Field(default="spectrum.dat", max_length=200)
    raw_text: str = Field(max_length=2_000_000)
    mapping: Mapping = Field(default_factory=Mapping)
    parameters: Parameters = Field(default_factory=Parameters)
    sample: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=5000)
