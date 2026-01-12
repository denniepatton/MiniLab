"""
Figure Generation Tool.

Provides figure generation and visualization capabilities:
- fig.create: Create figure from data
- fig.save: Save figure to file
- fig.render: Render figure specification

Uses matplotlib for rendering with support for common plot types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union, Literal
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..security import PathGuard, AccessDenied
from ..utils import console


PlotType = Literal[
    "line", "scatter", "bar", "barh", "hist",
    "box", "violin", "heatmap", "pie",
    "area", "density", "hexbin"
]


class CreateFigureInput(ToolInput):
    """Input for creating a figure."""
    data: dict[str, Any] = Field(..., description="Data to plot (dict or path to CSV)")
    plot_type: PlotType = Field("line", description="Type of plot")
    x: Optional[str] = Field(None, description="X-axis column/key")
    y: Optional[str] = Field(None, description="Y-axis column/key")
    title: str = Field("", description="Figure title")
    xlabel: str = Field("", description="X-axis label")
    ylabel: str = Field("", description="Y-axis label")
    figsize: tuple[float, float] = Field((10, 6), description="Figure size (width, height)")
    style: str = Field("seaborn-v0_8-whitegrid", description="Matplotlib style")
    color: Optional[str] = Field(None, description="Plot color")
    colormap: str = Field("viridis", description="Colormap for heatmaps")


class SaveFigureInput(ToolInput):
    """Input for saving a figure."""
    output_path: str = Field(..., description="Path for the output figure")
    format: str = Field("png", description="Output format (png, pdf, svg, jpg)")
    dpi: int = Field(150, description="Resolution in DPI")


class RenderSpecInput(ToolInput):
    """Input for rendering a figure from specification."""
    spec: dict[str, Any] = Field(..., description="Figure specification dict")
    output_path: str = Field(..., description="Path for the output figure")
    format: str = Field("png", description="Output format")


class FigureOutput(ToolOutput):
    """Output for figure operations."""
    path: Optional[str] = None
    figure_id: Optional[str] = None
    size_bytes: Optional[int] = None


class FigureTool(Tool):
    """
    Tool for creating and saving figures.
    
    Supports common plot types:
    - Line, scatter, bar charts
    - Histograms, box plots, violin plots
    - Heatmaps
    - Area plots
    
    Uses matplotlib with seaborn styling by default.
    """
    
    name = "figure"
    description = "Create and save figures/plots"
    
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._current_figure = None
        self._figure_counter = 0
    
    def get_actions(self) -> dict[str, str]:
        return {
            "create": "Create a figure from data",
            "save": "Save the current figure to file",
            "render": "Render a figure from JSON specification",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "create": CreateFigureInput,
            "save": SaveFigureInput,
            "render": RenderSpecInput,
        }
        return schemas.get(action, ToolInput)
    
    async def execute(self, action: str, params: dict[str, Any]) -> ToolOutput:
        """Execute a figure action."""
        schema = self.get_input_schema(action)
        validated = schema(**params)
        
        try:
            if action == "create":
                return await self._create_figure(validated)
            elif action == "save":
                return await self._save_figure(validated)
            elif action == "render":
                return await self._render_spec(validated)
            else:
                return FigureOutput(
                    success=False,
                    error=f"Unknown action: {action}",
                )
        except Exception as e:
            return FigureOutput(success=False, error=str(e))
    
    async def _create_figure(self, params: CreateFigureInput) -> FigureOutput:
        """Create a figure from data."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return FigureOutput(
                success=False,
                error="matplotlib not installed. Install with: pip install matplotlib",
            )
        
        # Try to set style
        try:
            plt.style.use(params.style)
        except Exception:
            plt.style.use('default')
        
        # Load data
        data = params.data
        if isinstance(data, str):
            # It's a path - load it
            data_path = Path(data)
            if data_path.suffix == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(data_path)
                    data = df.to_dict('list')
                except ImportError:
                    return FigureOutput(
                        success=False,
                        error="pandas required for CSV loading",
                    )
            elif data_path.suffix == '.json':
                with open(data_path) as f:
                    data = json.load(f)
        
        # Create figure
        fig, ax = plt.subplots(figsize=params.figsize)
        
        try:
            x_data = data.get(params.x) if params.x else None
            y_data = data.get(params.y) if params.y else None
            
            if params.plot_type == "line":
                if x_data is not None and y_data is not None:
                    ax.plot(x_data, y_data, color=params.color)
                elif y_data is not None:
                    ax.plot(y_data, color=params.color)
                    
            elif params.plot_type == "scatter":
                if x_data is not None and y_data is not None:
                    ax.scatter(x_data, y_data, color=params.color, alpha=0.7)
                    
            elif params.plot_type == "bar":
                if x_data is not None and y_data is not None:
                    ax.bar(x_data, y_data, color=params.color)
                    
            elif params.plot_type == "barh":
                if x_data is not None and y_data is not None:
                    ax.barh(x_data, y_data, color=params.color)
                    
            elif params.plot_type == "hist":
                hist_data = y_data if y_data else (x_data if x_data else list(data.values())[0])
                ax.hist(hist_data, color=params.color, edgecolor='white', alpha=0.7)
                
            elif params.plot_type == "box":
                box_data = [v for v in data.values() if isinstance(v, list)]
                ax.boxplot(box_data, labels=list(data.keys()))
                
            elif params.plot_type == "heatmap":
                # Assume data is 2D array or dict of lists
                if isinstance(data, dict):
                    import pandas as pd
                    df = pd.DataFrame(data)
                    heatmap_data = df.values
                else:
                    heatmap_data = np.array(data)
                im = ax.imshow(heatmap_data, cmap=params.colormap, aspect='auto')
                plt.colorbar(im, ax=ax)
                
            elif params.plot_type == "pie":
                values = y_data if y_data else list(data.values())[0]
                labels = x_data if x_data else list(data.keys())
                ax.pie(values, labels=labels, autopct='%1.1f%%')
                
            elif params.plot_type == "area":
                if x_data is not None and y_data is not None:
                    ax.fill_between(x_data, y_data, alpha=0.5, color=params.color)
                    ax.plot(x_data, y_data, color=params.color)
            
            # Labels and title
            if params.title:
                ax.set_title(params.title, fontsize=14, fontweight='bold')
            if params.xlabel:
                ax.set_xlabel(params.xlabel)
            if params.ylabel:
                ax.set_ylabel(params.ylabel)
            
            plt.tight_layout()
            
        except Exception as e:
            plt.close(fig)
            return FigureOutput(
                success=False,
                error=f"Error creating plot: {e}",
            )
        
        # Store figure
        self._figure_counter += 1
        figure_id = f"fig_{self._figure_counter}"
        self._current_figure = fig
        
        return FigureOutput(
            success=True,
            figure_id=figure_id,
            data={"message": f"Figure created: {figure_id}"},
        )
    
    async def _save_figure(self, params: SaveFigureInput) -> FigureOutput:
        """Save the current figure to file."""
        if self._current_figure is None:
            return FigureOutput(
                success=False,
                error="No figure to save. Create a figure first.",
            )
        
        from ..security import PathGuard
        
        output_path = Path(params.output_path)
        
        # Check write permission
        guard = PathGuard.get_instance()
        if not guard.can_write(output_path):
            raise AccessDenied(str(output_path), "write")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._current_figure.savefig(
                str(output_path),
                format=params.format,
                dpi=params.dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
            )
            
            import matplotlib.pyplot as plt
            plt.close(self._current_figure)
            self._current_figure = None
            
            size = output_path.stat().st_size
            
            return FigureOutput(
                success=True,
                path=str(output_path),
                size_bytes=size,
            )
            
        except Exception as e:
            return FigureOutput(
                success=False,
                error=f"Error saving figure: {e}",
            )
    
    async def _render_spec(self, params: RenderSpecInput) -> FigureOutput:
        """Render a figure from JSON specification."""
        spec = params.spec
        
        # Create figure from spec
        create_params = CreateFigureInput(
            data=spec.get("data", {}),
            plot_type=spec.get("plot_type", "line"),
            x=spec.get("x"),
            y=spec.get("y"),
            title=spec.get("title", ""),
            xlabel=spec.get("xlabel", ""),
            ylabel=spec.get("ylabel", ""),
            figsize=tuple(spec.get("figsize", [10, 6])),
            style=spec.get("style", "seaborn-v0_8-whitegrid"),
            color=spec.get("color"),
            colormap=spec.get("colormap", "viridis"),
        )
        
        result = await self._create_figure(create_params)
        if not result.success:
            return result
        
        # Save the figure
        save_params = SaveFigureInput(
            output_path=params.output_path,
            format=params.format,
            dpi=spec.get("dpi", 150),
        )
        
        return await self._save_figure(save_params)
