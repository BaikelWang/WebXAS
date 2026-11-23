"""Portable display contracts, independent of processing and fit settings."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FigureStyle(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    profile: Literal['classic', 'nature'] = Field(default='classic', exclude_if=lambda value: value == 'classic')
    layout: Literal['single', 'double', 'presentation'] = 'single'
    dpi: Literal[300, 600] = 300
    # Retained for old projects/API payloads; plot text is always English.
    language: Literal['zh', 'en'] = 'zh'
    formats: list[Literal['svg', 'pdf', 'png', 'tiff']] = Field(default_factory=lambda: ['svg', 'pdf', 'png', 'tiff'])
    palette: Literal['okabe_ito', 'muted'] = 'okabe_ito'
    font_size: float | None = Field(default=None, ge=5, le=24)
    linewidth: float = Field(default=1.1, ge=.3, le=4)
    legend: bool = True
    offset: float = Field(default=0, ge=-1000, le=1000)
    xlim: list[float] | None = None
    ylim: list[float] | None = None
    transparent: bool = False

    @model_validator(mode='after')
    def ranges(self):
        if not self.formats or len(self.formats) != len(set(self.formats)):
            raise ValueError('Select distinct export formats.')
        # An always-present PNG is the actual preview consumed by the UI.
        if 'png' not in self.formats:
            raise ValueError('PNG is required for the offline and application preview.')
        if self.profile == 'nature':
            if self.layout == 'presentation' or (self.font_size is not None and not 5 <= self.font_size <= 7):
                raise ValueError('Nature artwork uses single or double columns and 5–7 pt lettering.')
            if self.linewidth > 1:
                raise ValueError('Nature artwork requires strokes at most 1 pt.')
        elif self.font_size is not None and self.font_size < 8:
            raise ValueError('Classic lettering must be at least 8 pt.')
        for limits in (self.xlim, self.ylim):
            if limits is not None and (len(limits) != 2 or limits[0] >= limits[1]):
                raise ValueError('Display limits require ordered lower and upper values.')
        return self

    @property
    def size_mm(self):
        if self.profile == 'nature':
            return (89., 67.) if self.layout == 'single' else (183., 105.)
        return {'single': (85., 68.), 'double': (180., 115.),
                'presentation': (254., 142.875)}[self.layout]

    @property
    def point_size(self):
        return self.font_size or (6.5 if self.profile == 'nature' else 15. if self.layout == 'presentation' else 9.)

    @property
    def render_language(self):
        return 'en'

    @property
    def font_family(self):
        if self.profile == 'nature':
            from .scientific_style import nature_font
            return nature_font()
        return 'Times New Roman'


class FigureSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    version: Literal[1] = 1
    template_version: str = 'webxas-scientific-1'
    id: str
    family: str
    title: str
    title_en: str
    kind: Literal['line', 'residual', 'heatmap', 'composite', 'trend', 'geometry'] = 'line'
    sources: list[dict] = Field(default_factory=list)
    panels: list[dict] = Field(default_factory=list)
    transforms: dict = Field(default_factory=dict)
    method: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    data_files: list[str] = Field(default_factory=list)
    panel_id: str | None = None
    parent_figure_id: str | None = None
    panel_index: int | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    sample_names: list[str] = Field(default_factory=list)
    insight: dict = Field(default_factory=dict)
    style: FigureStyle = Field(default_factory=FigureStyle)
