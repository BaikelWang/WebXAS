"""Snapshot-bound scientific figures; presentation never edits scientific arrays."""
from .spec import FigureSpec, FigureStyle
from .sources import collect_figures
from .render import render_figure

__all__ = ['FigureSpec', 'FigureStyle', 'collect_figures', 'render_figure']
