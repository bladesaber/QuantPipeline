import pandas as pd
import numpy as np
from typing import List, Literal, Any
from abc import ABC, abstractmethod

import bokeh.plotting as plt
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Turbo256, Category20c, Viridis256
from bokeh.transform import linear_cmap
from bokeh.models import HoverTool, ColumnDataSource, LinearAxis, Range1d, ColorBar, BasicTicker, DataTable, TableColumn, RangeTool
from bokeh.io import curdoc
from colorama import Fore, Back, Style, init


init(autoreset=True)


class BokehItem(ABC):
    def __init__(self, name: str):
        self.name = name
        self.relate_figures = set()


class BokehSeries(BokehItem):
    def __init__(self, name: str, data: np.ndarray | list[str], labels: list[str] = None, priority: np.ndarray = None):
        super().__init__(name)
        self.data = data
        self.labels = labels
        self.priority = priority


class BokehMatrix(BokehItem):
    def __init__(self, name: str, data: np.ndarray, x_label: str, y_label: str):
        super().__init__(name)
        self.data = data
        self.x_label = x_label
        self.y_label = y_label


class BokehTable(BokehItem):
    def __init__(self, name: str, data: pd.DataFrame, with_index: bool=True):
        super().__init__(name)
        self.data = data
        self.with_index = with_index


class BokehStockBar(BokehItem):
    def __init__(self, name: str, data: pd.DataFrame):
        super().__init__(name)
        self.data = data


class BokehAxis(object):
    def __init__(self, name: str, axis_min: float, axis_max: float):
        self.name = name
        self.axis_min = axis_min
        self.axis_max = axis_max
        self.axis_range = Range1d(start=axis_min - (axis_max - axis_min) * 0.1, end=axis_max + (axis_max - axis_min) * 0.1)
        self.relate_figures = set()


class BokehFigure(ABC):    
    def __init__(self, title: str, width: int = None, height: int = None):
        self.title = title
        self.width = width
        self.height = height
        self.figure = None
    
    @abstractmethod
    def postprocess(self):
        raise NotImplementedError('BokehFigure.postprocess must be implemented')


class BokehBasicFigure(BokehFigure):
    def __init__(
        self, title: str, width: int = None, height: int = None,
        legend_location: str = "top_right", legend_click_policy: str = "hide",
        tools: list[str] = None,
    ):
        super().__init__(title, width, height)
        if tools is None:
            self.tools = ["box_select", "reset", "pan", "wheel_zoom"]
        else:
            self.tools = tools
        self.legend_location = legend_location
        self.legend_click_policy = legend_click_policy
        
        self.figure = plt.figure(title=title, tools=self.tools, width=width, height=height)
        self.figure.extra_y_ranges = {}
        
        self.x_data: BokehSeries = None
        self.ys_data: dict[str, dict[str, BokehSeries | float | str | int]] = {}
        self.hover_tools = []
    
    def draw_x(self, x_data: BokehSeries):
        self.x_data = x_data
        self.figure.xaxis.axis_label = x_data.name
        self.x_data.relate_figures.add(self)
    
    def draw_line(self, y_data: BokehSeries, color: List[float], y_axis: BokehAxis, line_width: int = 2):
        assert y_axis.name not in self.ys_data, f"{Fore.RED}[Error] y_axis.name must be unique"
        assert self.x_data is not None, f"{Fore.RED}[Error] x_data must be set before drawing line"
        
        y_data.relate_figures.add(self)
        self.figure.extra_y_ranges.setdefault(y_axis.name, y_axis.axis_range)
        y_axis.relate_figures.add(self)
        
        source_dict = {
            "x": self.x_data.data, "y": y_data.data,
        }
        if y_data.labels is not None:
            source_dict["label"] = y_data.labels
        data_source = ColumnDataSource(source_dict)
        
        line_element = self.figure.line(
            x="x", y="y", source=data_source, y_range_name=y_axis.name, legend_label=y_data.name,
            color=color, line_width=line_width, 
        )
        self.ys_data[y_data.name] = {"data": y_data, "mode": "line", "ui_element": line_element, 'source': data_source}
    
    def draw_scatter(
        self, y_data: BokehSeries, color: List[float], size: int = 2, marker: str = "circle", transparent: float = 0.6, with_hover: bool = False
    ):
        assert y_data.name not in self.ys_data, f"{Fore.RED}[Error] y_data.name must be unique"
        
        y_data.relate_figures.add(self)
        source_dict = {
            "x": self.x_data.data,
            "y": y_data.data,
        }
        if y_data.labels is not None:
            source_dict["label"] = y_data.labels
        data_source = ColumnDataSource(source_dict)
        
        if y_data.priority is None:
            scatter_element = self.figure.scatter(
                x="x", y="y", source=data_source, legend_label=y_data.name,
                color=color, size=size, marker=marker, fill_color=color, fill_alpha=transparent
            )
        else:
            mapper = linear_cmap(field_name="y", palette=Turbo256, low=min(y_data.data), high=max(y_data.data))
            scatter_element = self.figure.scatter(
                x="x", y="y", source=data_source, legend_label=y_data.name, color=mapper, 
                size=size, marker=marker, fill_alpha=transparent,
            )
        self.ys_data[y_data.name] = {"data": y_data, "mode": "scatter", "ui_element": scatter_element, 'source': data_source}
        
        if with_hover and y_data.labels is not None:
            hover = HoverTool(
                renderers=[scatter_element],
                tooltips=[
                    (f"{y_data.name}", "@label"),
                    ("(X,Y)", "@x{0.00}, @y{0.00}")
                ],
                mode='mouse'
            )
            self.hover_tools.append(hover)
        
    def draw_bars(self, y_data: BokehSeries, color: List[float], width: int = 2, with_hover: bool = False):
        assert y_data.name not in self.ys_data, f"{Fore.RED}[Error] y_data.name must be unique"
        
        y_data.relate_figures.add(self)
        source_dict = {
            "x": self.x_data.data,
            "y": y_data.data,
        }
        if y_data.labels is not None:
            source_dict["label"] = y_data.labels
        data_source = ColumnDataSource(source_dict)
        bars_element = self.figure.vbar(
            x="x", top="y", bottom=0, source=data_source, legend_label=y_data.name,
            color=color, width=width
        )
        self.ys_data[y_data.name] = {"data": y_data, "mode": "bars", "ui_element": bars_element, 'source': data_source}
        
        if with_hover and y_data.labels is not None:
            hover = HoverTool(
                renderers=[bars_element],
                tooltips=[
                    (f"{y_data.name}", "@label"),
                    ("(X,Y)", "@x{0.00}, @y{0.00}")
                ],
                mode='mouse'
            )
            self.hover_tools.append(hover)
    
    def draw_pie(self, y_data: BokehSeries, radius: float = 1):
        assert y_data.name not in self.ys_data, f"{Fore.RED}[Error] y_data.name must be unique"
        assert len(self.ys_data) == 0, f"{Fore.RED}[Error] Only one y_data is allowed for pie chart"
        
        total_value = np.sum(y_data.data)
        percent_values = y_data.data / total_value
        angles = percent_values * 2 * np.pi
        start_angles = np.cumsum([0] + angles[:-1].tolist())
        end_angles = np.cumsum(angles)
        
        data_source = ColumnDataSource({
            "categories": self.x_data.data,
            "start_angle": start_angles,
            "end_angle": end_angles,
            "color": Category20c[len(y_data.data)]
        })
        pie_element = self.figure.wedge(
            x=0, y=0, radius=radius, start_angle="start_angle", end_angle="end_angle", 
            legend_label=y_data.name, legend_field="categories", source=data_source,
            color="color", line_color="black",
        )
        self.ys_data[y_data.name] = {"data": y_data, "mode": "pie", "ui_element": pie_element, 'source': data_source}
        
        self.figure.x_range = Range1d(radius - 0.5, radius + 0.5)
        self.figure.y_range = Range1d(radius - 0.5, radius + 0.5)
    
    def postprocess(self):
        if len(self.hover_tools) > 0:
            self.figure.add_tools(*self.hover_tools)
        
        self.figure.legend.location = self.legend_location
        self.figure.legend.click_policy = self.legend_click_policy
        
        for name in self.figure.extra_y_ranges.keys():
            self.figure.add_layout(LinearAxis(y_range_name=name, axis_label=name), 'right')


class BokehMapFigure(BokehFigure):
    def __init__(self, title: str, width: int = None, height: int = None, tools: list[str] = None):
        super().__init__(title, width, height)
        if tools is None:
            self.tools = ["box_select", "reset", "pan", "wheel_zoom"]
        else:
            self.tools = tools
        self.figure = plt.figure(title=title, tools=self.tools, width=width, height=height)
        
        self.map_data: dict[str, BokehMatrix] = {}
        
    def draw_heatmap(self, data: BokehMatrix, x_range: tuple[float, float], y_range: tuple[float, float]):
        self.map_data[data.name] = data
        data.relate_figures.add(self)
        
        xs = np.linspace(x_range[0], x_range[1], data.data.shape[1])
        ys = np.linspace(y_range[0], y_range[1], data.data.shape[0])
        xx, yy = np.meshgrid(xs, ys)
        
        data_source = ColumnDataSource(data={
            "x": xx.ravel(), "y": yy.ravel(), "value": data.data.ravel()
        })
        mapper = linear_cmap(field_name="value", palette=Viridis256, low=min(data.data), high=max(data.data))

        map_element = self.figure.rect(
            x="x", y="y", source=data_source, fill_color = {'field': 'value', 'transform': mapper},
            width=1, height=1, line_color=None
        )
        self.map_data[data.name] = {"mode": "heatmap", "data": data, "ui_element": map_element, 'source': data_source}
    
    def postprocess(self):
        return


class BokehTableFigure(BokehFigure):
    def __init__(self, title: str, width: int = None, height: int = None):
        super().__init__(title, width, height)
        self.table_data: dict[str, BokehTable] = {}
    
    def draw_table(self, data: BokehTable):
        self.table_data[data.name] = data
        data.relate_figures.add(self)
        
        columns = []
        if data.with_index:
            index_name = data.data.index.name
            columns.append(TableColumn(field=index_name, title=index_name))
            source = ColumnDataSource(data.data.reset_index())
        else:
            source = ColumnDataSource(data.data)
        columns.extend([TableColumn(field=col, title=col) for col in data.data.columns])        
        self.figure = DataTable(columns=columns, source=source, width=self.width, height=self.height)
    
    def postprocess(self):
        return


class BokehStockBarFigure(BokehFigure):
    def __init__(
        self, title: str, figure_width: int = None, price_height: int = None, volume_height: int = None,
        legend_location: str = "top_right", legend_click_policy: str = "hide",
        tools: list[str] = None,
    ):
        super().__init__(title, figure_width, price_height + volume_height)
        if tools is None:
            self.tools = ["box_select", "reset", "pan", "wheel_zoom"]
        else:
            self.tools = tools
        self.legend_location = legend_location
        self.legend_click_policy = legend_click_policy
        
        self.figure_width = figure_width
        self.price_height = price_height
        self.volume_height = volume_height
        self.price_figure = plt.figure(title=title, tools=self.tools, width=self.figure_width, height=self.price_height)
        self.price_figure.extra_y_ranges = {}
        self.price_figure.grid.grid_line_alpha = 0.3
        self.vol_figures: list[plt.figure] = []
        
        self.bars_data: dict[str, BokehStockBar] = {}
        self.hover_tools: list[HoverTool] = []
        
        self.select_header: plt.figure = None
        self.range_tool: RangeTool = None
        self.figure: column = None

    def draw_stock_bar(self, data: BokehStockBar, y_axis: BokehAxis, with_volume: bool = False):
        self.bars_data.setdefault(data.name, data)
        data.relate_figures.add(self)
        
        self.price_figure.extra_y_ranges.setdefault(y_axis.name, y_axis.axis_range)
        y_axis.relate_figures.add(self)
        
        _df = pd.DataFrame(data.data)
        _df['color'] = np.where(_df['close'] > _df['open'], 'green', 'red')
        source = ColumnDataSource(_df)
        
        segment_element = self.price_figure.segment(
            x0='date_time', x1='date_time', y0='low', y1='high', source=source, y_range_name=y_axis.name, legend_label=data.name,
            color="black", line_width=2,
        )
        bar_element = self.price_figure.vbar(
            x='date_time', width=0.5, top='open', bottom='close', source=source, y_range_name=y_axis.name, legend_label=data.name,
            fill_color='color', line_color="black",
        )
        
        self.bars_data[data.name] = {
            "mode": "stock_bar", "data": data, "segment_element": segment_element, "bar_element": bar_element,
            "price_source": source
        }
        
        if with_volume:
            # ------ already linked to price_figure.x_range
            vol_figure = plt.figure(
                x_axis_type="datetime", x_range=self.price_figure.x_range, width=self.figure_width, height=self.volume_height
            )
            vol_figure.vbar(x='date_time', width=0.5, top='volume', bottom=0, color='color', source=source)
            vol_figure.yaxis.axis_label = "Volume"
            self.vol_figures.append(vol_figure)
            self.bars_data[data.name].update({"vol_element": vol_figure, "vol_source": source})
            
    def postprocess(self):
        self.price_figure.legend.location = self.legend_location
        self.price_figure.legend.click_policy = self.legend_click_policy
        for name in self.price_figure.extra_y_ranges.keys():
            self.price_figure.add_layout(LinearAxis(y_range_name=name, axis_label=name), 'right')

        self.select_header = plt.figure(
            x_axis_type="datetime", y_axis_type=None, tools="", toolbar_location=None, width=self.figure_width, height=30
        )
        self.select_header.xaxis.visible = False
        self.select_header.yaxis.visible = False
        self.select_header.line(
            'date_time', 'close', source=self.bars_data[list(self.bars_data.keys())[0]]['price_source'],
            color="black", line_width=2
        )
        
        self.range_tool = RangeTool(x_range=self.price_figure.x_range)
        self.range_tool.overlay.fill_color = "navy"
        self.range_tool.overlay.fill_alpha = 0.2
        self.select_header.add_tools(self.range_tool)
        
        self.figure = column(self.select_header, self.price_figure, *self.vol_figures)


class BokehGridLayout(object):
    def __init__(self, nrows: int, ncols: int):
        self.nrows = nrows
        self.ncols = ncols
        self.layout: list[list[BokehFigure]] = [[None for _ in range(ncols)] for _ in range(nrows)]
        self.figure_dict: dict[str, BokehFigure] = {}
    
    def add_figure(self, figure: BokehFigure, row: int, col: int):
        self.layout[row][col] = figure.figure
        self.figure_dict[figure.title] = figure
        
    def get_layout(self):
        layout = []
        for row in range(self.nrows):
            if len(self.layout[row]) > 0:
                layout.append([])
            for col in range(self.ncols):
                if self.layout[row][col] is not None:
                    layout[-1].append(self.layout[row][col].figure)
        return gridplot(layout)
    
    @staticmethod
    def in_row(children: List[BokehFigure], sizing_mode: Literal["stretch_both", "fixed"] = "stretch_both"):
        return row([child.figure for child in children], sizing_mode=sizing_mode)
    
    @staticmethod
    def in_column(children: List[BokehFigure], sizing_mode: Literal["stretch_both", "fixed"] = "stretch_both"):
        return column([child.figure for child in children], sizing_mode=sizing_mode)
    
    @staticmethod
    def in_grid(children: List[BokehFigure], nrows: int, ncols: int):
        assert ncols * nrows >= len(children), f"{Fore.RED}[Error] ncols * nrows must be greater than or equal to the number of children"
        return gridplot([child.figure for child in children], nrows=nrows, ncols=ncols)
    
    def show(self, layout: Any, use_curdoc: bool = False, title: str = "Bokeh App"):
        """If using jave script, must use use_curdoc=True for script callback"""
        for figure in self.figure_dict.values():
            figure.postprocess()
        
        layout = self.get_layout()
        if use_curdoc:
            curdoc().add_root(layout)
            curdoc().title = title
        else:
            plt.show(layout)