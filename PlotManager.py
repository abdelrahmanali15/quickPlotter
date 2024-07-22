
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import mpl_toolkits.axisartist as axisartist
from qbstyles import mpl_style
# Constants


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        cv = self.canvas
        fig = cv.figure
        if self._bg is None:
            self.on_draw(None)
        else:
            cv.restore_region(self._bg)
            self._draw_animated()
            cv.blit(fig.bbox)
        cv.flush_events()


class PlotManager:
    def __init__(self, style='default', figsize=None):
        self.set_style(style)
        self.figsize = figsize
        self.fig = plt.figure(figsize=self.figsize)
        self.dark_mode = False
        self.axes = []
        self.current_ax = self.fig.add_subplot(111)
        self.axes.append(self.current_ax)
        self.setup_event_handlers()
        self.intersection_annotations = []
        self.dots = []
        self.annotations = []
        self.horizontals = []
        self.verticals = []
        self.intersection_dots = []
        self.cursor_id = 0
        self.a_cursor = None
        self.b_cursor = None
        self.diff_annotations = {}
        self.legends = {}

        # Initialize BlitManager
        self.blit_manager = BlitManager(self.fig.canvas)

    def format_axis(self, axes):
        for each_axes in axes:
            each_axes.yaxis.set_major_formatter(
                EngFormatter(unit="", useMathText=True, sep=""))
            each_axes.xaxis.set_major_formatter(
                EngFormatter(unit="", useMathText=True, sep=""))

    def set_style(self, style):
        if style == 'dark':
            plt.style.use(
                ['dark_background', 'pitayasmoothie-dark.mplstyle'])
            self.dark_mode = True
        elif style == 'light':
            plt.style.use(
                ['pitayasmoothie-light.mplstyle'])
        elif style == 'pacoty':
            plt.style.use(
                'pacoty.mplstyle')
        elif style == 'dark2':
            # mpl_style(dark=True)
            plt.style.use(
                'dark2.mplstyle')
            self.dark_mode = True
        elif style == 'light2':
            mpl_style(dark=False)
        elif style == 'prof':
            plt.style.use(
                ['computermodernstyle.mplstyle'])
        else:
            plt.style.use(style)


# --------------- Setup Events -------------#


    def setup_event_handlers(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)

    def on_canvas_click(self, event):
        self.current_ax = event.inaxes
        if self.fig.canvas.toolbar.mode != "" or event.button != 2:  # Check for middle click

            return

        if event.inaxes is None:
            return

        x, y = event.xdata, event.ydata
        print(f"x={x}, y={y}")

        dot = event.inaxes.plot(x, y, "ro", markersize=8)[0]
        self.dots.append(dot)

        formatter = EngFormatter(sep="")
        x_eng, y_eng = formatter(x), formatter(y)
        annotation = event.inaxes.annotate(
            f"({x_eng}, {y_eng})",
            (x, y),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue",
                      ec="b", lw=0.5, alpha=0.7),
            xytext=(0, 10),
            ha="center",
        )
        self.annotations.append(annotation)
        # self.fig.canvas.draw()
        self.blit_manager.add_artist(annotation)
        self.blit_manager.add_artist(dot)
        self.blit_manager.update()
        # self.blit_manager.on_draw(None)

    def on_key_press(self, event):
        if event.key == 'd':
            self.clear_annotations_and_dots()
        elif event.key == 'v':
            x = event.xdata if event.xdata is not None else 0
            self.add_vertical_cursor(x)
        elif event.key == 'V':
            self.remove_cursor(self.verticals)
        elif event.key == 'h':
            y = event.ydata if event.ydata is not None else 0
            self.add_horizontal_cursor(y)
        elif event.key == 'H':
            self.remove_cursor(self.horizontals)
        elif event.key == 'a':
            self.add_a_cursor(event.xdata)
        elif event.key == 'A':
            self.remove_a_cursor()
        elif event.key == 'b':
            self.add_b_cursor(event.xdata)
        elif event.key == 'B':
            self.remove_b_cursor()
        elif event.key == 'Q':
            self.remove_b_cursor()
            self.remove_a_cursor()
            self.delete_all_cursors()
            # self.remove_cursor(self.verticals)
            self.clear_annotations_and_dots()
        elif event.key == 'D':
            self.clear_plot(self.current_ax)

        elif event.key == 'F':
            self.delete_all_cursors()
            self.remove_axis()
            self.autoSize()

        self.autoSize()

    def on_pick(self, event):
        artist = event.artist
        if isinstance(artist, Line2D) and hasattr(artist, 'cursor_id'):
            cursor_lines = next(lines for cid, lines in self.horizontals +
                                self.verticals if cid == artist.cursor_id)

            def on_mouse_move(event):
                if event.inaxes is None:
                    return
                if artist.is_vertical:
                    self.update_cursor_position(cursor_lines, event.xdata)
                else:
                    self.update_cursor_position(cursor_lines, event.ydata)
                if hasattr(self, 'a_cursor') and hasattr(self, 'b_cursor'):
                    if artist.cursor_id in [self.a_cursor, self.b_cursor]:
                        self.update_diff_annotations()

            def on_release(event):
                self.fig.canvas.mpl_disconnect(self.cid_mouse_move)
                self.fig.canvas.mpl_disconnect(self.cid_release)

            self.cid_mouse_move = self.fig.canvas.mpl_connect(
                'motion_notify_event', on_mouse_move)
            self.cid_release = self.fig.canvas.mpl_connect(
                'button_release_event', on_release)

    # --------------- Cursors -------------#

    def add_a_cursor(self, x):
        if self.a_cursor:
            self.remove_a_cursor()
        self.a_cursor = self.add_vertical_cursor(x, 'red')
        self.update_diff_annotations()

    def add_b_cursor(self, x):
        if self.b_cursor:
            self.remove_b_cursor()
        self.b_cursor = self.add_vertical_cursor(x, 'orange')
        self.update_diff_annotations()

    def remove_a_cursor(self):
        if self.a_cursor:
            self.remove_cursor_by_id(self.a_cursor)
            self.a_cursor = None
        self.update_diff_annotations()

    def remove_b_cursor(self):
        if self.b_cursor:
            self.remove_cursor_by_id(self.b_cursor)
            self.b_cursor = None
        self.update_diff_annotations()

    def remove_cursor_by_id(self, cursor_id):
        for cursor_list in [self.verticals, self.horizontals]:
            for i, (cid, lines) in enumerate(cursor_list):
                if cid == cursor_id:
                    for line in lines:
                        line.remove()
                        self.blit_manager._artists.remove(line)
                    cursor_list.pop(i)
                    break
        self.blit_manager.update()

    def delete_all_cursors(self):
        if self.horizontals:
            while self.horizontals:
                self.remove_cursor(self.horizontals)
            self.horizontals.clear()  # Clear the list after removal
        if self.verticals:
            while self.verticals:
                self.remove_cursor(self.verticals)
            self.verticals.clear()  # Clear the list after removal

    def remove_cursor(self, cursor_list):
        if cursor_list:
            _, lines = cursor_list.pop()
            for line in lines:
                line.remove()
                self.blit_manager._artists.remove(line)
            self.update_intersection_dots_and_annotations()
            # self.fig.canvas.draw()
            self.blit_manager.update()

    def add_vertical_cursor(self, x, color='#474747', annotate_indices='*'):
        cursor_id = self.get_next_cursor_id()
        cursor_lines = []
        for ax in self.axes:
            v_line = ax.axvline(
                x=x, color=color, linestyle='--', picker=True, lw=1)
            v_line.cursor_id = cursor_id
            v_line.is_vertical = True
            v_line.annotate_indices = annotate_indices
            cursor_lines.append(v_line)
            self.blit_manager.add_artist(v_line)
        self.verticals.append((cursor_id, cursor_lines))
        self.update_intersection_dots_and_annotations()
        self.blit_manager.update()
        return cursor_id

    def add_horizontal_cursor(self, y, color='#474747', annotate_indices='*'):
        cursor_id = self.get_next_cursor_id()
        cursor_lines = []
        for ax in self.axes:
            h_line = ax.axhline(
                y=y, color=color, linestyle='--', picker=True, lw=1)
            h_line.cursor_id = cursor_id
            h_line.is_vertical = False
            h_line.annotate_indices = annotate_indices
            cursor_lines.append(h_line)
            self.blit_manager.add_artist(h_line)
        self.horizontals.append((cursor_id, cursor_lines))
        self.update_intersection_dots_and_annotations()
        self.blit_manager.update()
        return cursor_id

    def get_next_cursor_id(self):
        self.cursor_id += 1
        return self.cursor_id

    def update_cursor_position(self, cursor_lines, value):
        is_vertical = cursor_lines[0].is_vertical
        for line in cursor_lines:
            if is_vertical:
                line.set_xdata([value, value])
            else:
                line.set_ydata([value, value])
        self.update_intersection_dots_and_annotations()
        self.autoSize()

    def add_vertical_cursor_at(self, x, annotate_indices='*'):
        """
        Add a vertical cursor at a specific x position.

        Parameters:
        x (float): The x-coordinate where the vertical cursor should be placed.
        annotate_indices (str|int|list): Indices of intersections to annotate.
                                         '*' for all, int for single index, list for multiple indices.

        Returns:
        int: The cursor ID of the newly created cursor.
        """
        return self.add_vertical_cursor(x, annotate_indices=annotate_indices)

    def add_horizontal_cursor_at(self, y, annotate_indices='*'):
        """
        Add a horizontal cursor at a specific y position.

        Parameters:
        y (float): The y-coordinate where the horizontal cursor should be placed.
        annotate_indices (str|int|list): Indices of intersections to annotate.
                                         '*' for all, int for single index, list for multiple indices.

        Returns:
        int: The cursor ID of the newly created cursor.
        """
        return self.add_horizontal_cursor(y, annotate_indices=annotate_indices)

# --------------- Intersections -------------#

    def find_intersections(self, x_data, y_data, value, is_horizontal):
        if is_horizontal:
            # Find x values where the curve crosses the horizontal line
            signs = np.sign(y_data - value)
            zero_crossings = (signs[:-1] != signs[1:]) & (signs[:-1] != 0)
            indices = np.where(zero_crossings)[0]

            intersections = []
            for i in indices:
                x1, y1 = x_data[i], y_data[i]
                x2, y2 = x_data[i+1], y_data[i+1]
                if y1 != y2:
                    x_intersect = x1 + (x2 - x1) * (value - y1) / (y2 - y1)
                    intersections.append(x_intersect)

            return intersections
        else:
            # Find y values where the curve crosses the vertical line
            indices = np.where((x_data[:-1] <= value)
                               & (x_data[1:] > value))[0]

            intersections = []
            for i in indices:
                x1, y1 = x_data[i], y_data[i]
                x2, y2 = x_data[i+1], y_data[i+1]
                if x1 != x2:
                    y_intersect = y1 + (y2 - y1) * (value - x1) / (x2 - x1)
                    intersections.append(y_intersect)

            return intersections

    def get_intersection(self, ax, value, is_vertical):
        lines = [line for line in ax.get_lines(
        ) if not line.get_linestyle() == '--']
        if not lines:
            return None
        line = lines[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        if is_vertical:
            interp_func = interp1d(
                x_data, y_data, kind='linear', fill_value="extrapolate")
            y = interp_func(value)
            return value, y
        else:
            interp_func = interp1d(
                y_data, x_data, kind='linear', fill_value="extrapolate")
            x = interp_func(value)
            return x, value

    # --------------- Annotations -------------#

    def update_diff_annotations(self):
        for ax in self.axes:
            if ax in self.diff_annotations:
                self.diff_annotations[ax].remove()
                self.blit_manager._artists.remove(self.diff_annotations[ax])
                del self.diff_annotations[ax]

        if self.a_cursor and self.b_cursor:
            a_line = next(
                lines for cid, lines in self.verticals if cid == self.a_cursor)[0]
            b_line = next(
                lines for cid, lines in self.verticals if cid == self.b_cursor)[0]

            a_x = a_line.get_xdata()[0]
            b_x = b_line.get_xdata()[0]

            x_diff = abs(b_x - a_x)

            for i, ax in enumerate(self.axes):
                lines = [line for line in ax.get_lines()
                         if not line.get_linestyle() == '--' and not len(line.get_xdata()) < 6]

                if lines:
                    formatter = EngFormatter(sep="")
                    x_diff_eng = formatter(x_diff)

                    y_diffs = []
                    for line in lines:
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        a_y = np.interp(a_x, x_data, y_data)
                        b_y = np.interp(b_x, x_data, y_data)
                        y_diff = abs(b_y - a_y)
                        y_diffs.append(formatter(y_diff))

                    text = f"$Δx$: {
                        x_diff_eng}\n" + "\n".join(f"$Δy_{j+1}$: {y_diff}" for j, y_diff in enumerate(y_diffs))

                    self.diff_annotations[ax] = ax.annotate(text,
                                                            xy=(0.02, 0.98),
                                                            xycoords='axes fraction',
                                                            verticalalignment='top',
                                                            fontsize=10,
                                                            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="b", lw=0.5, alpha=0.5))
                    self.blit_manager.add_artist(self.diff_annotations[ax])

        self.blit_manager.update()

    def add_intersection_annotation(self, ax, x, y, formatter):
        x_eng, y_eng = formatter(x), formatter(y)
        if self.dark_mode:
            annotation = ax.annotate(
                f"({x_eng}, {y_eng})",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#0C1C23",
                          ec="#FFFFFF07", lw=0.5, alpha=1),
                zorder=5
            )
        else:
            annotation = ax.annotate(
                f"({x_eng}, {y_eng})",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFE0",
                          ec="#A9A9A9", lw=0.5, alpha=1),
                zorder=5)
            self.intersection_annotations.append(annotation)
        self.blit_manager.add_artist(annotation)

    def clear_annotations_and_dots(self):
        for dot in self.dots:
            dot.remove()
            self.blit_manager._artists.remove(dot)
        for annotation in self.annotations:
            annotation.remove()
            self.blit_manager._artists.remove(annotation)
        self.blit_manager.update()
        self.dots.clear()
        self.annotations.clear()

    def update_cursor_annotation(self, cursor, value, is_vertical):
        ax = cursor.axes
        if hasattr(cursor, 'annotation'):
            cursor.annotation.remove()

        intersection = self.get_intersection(ax, value, is_vertical)
        if intersection:
            x, y = intersection
            formatter = EngFormatter(sep="")
            x_eng, y_eng = formatter(x), formatter(y)
            text = f"({x_eng}, {y_eng})"
            if is_vertical:
                cursor.annotation = ax.annotate(text, (x, y), xytext=(5, 0),
                                                textcoords="offset points",
                                                va='bottom')
            else:
                cursor.annotation = ax.annotate(text, (x, y), xytext=(0, 5),
                                                textcoords="offset points",
                                                ha='right')

    def update_intersection_dots_and_annotations(self):
        for dot in self.intersection_dots:
            dot.remove()
            self.blit_manager._artists.remove(dot)
        self.intersection_dots.clear()

        for annotation in self.intersection_annotations:
            annotation.remove()
            self.blit_manager._artists.remove(annotation)
        self.intersection_annotations.clear()

        formatter = EngFormatter(sep="")

        for ax in self.axes:
            lines = [line for line in ax.get_lines(
            ) if not line.get_linestyle() == '--']

            for _, h_lines in self.horizontals:
                y = h_lines[0].get_ydata()[0]
                annotate_indices = h_lines[0].annotate_indices
                self.process_intersections(
                    ax, lines, y, True, annotate_indices, formatter)

            for _, v_lines in self.verticals:
                x = v_lines[0].get_xdata()[0]
                annotate_indices = v_lines[0].annotate_indices
                self.process_intersections(
                    ax, lines, x, False, annotate_indices, formatter)

        self.blit_manager.update()

    def process_intersections(self, ax, lines, value, is_horizontal, annotate_indices, formatter):
        intersections = []
        for i, line in enumerate(lines):
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            if is_horizontal:
                x_intersections = self.find_intersections(
                    x_data, y_data, value, is_horizontal=True)
                for x in x_intersections:
                    intersections.append((i, x, value))
            else:
                y_intersections = self.find_intersections(
                    x_data, y_data, value, is_horizontal=False)
                for y in y_intersections:
                    intersections.append((i, value, y))

        if annotate_indices == '*':
            indices_to_annotate = range(len(intersections))
        elif isinstance(annotate_indices, int):
            # Convert to 0-based index
            indices_to_annotate = [annotate_indices - 1]
        elif isinstance(annotate_indices, list):
            # Convert to 0-based indices
            indices_to_annotate = [i - 1 for i in annotate_indices]
        else:
            raise ValueError("Invalid annotate_indices value")

        for i in indices_to_annotate:
            if 0 <= i < len(intersections):
                _, x, y = intersections[i]
                dot = ax.plot(x, y, 'mo', markersize=8)[0]
                self.intersection_dots.append(dot)
                self.blit_manager.add_artist(dot)
                self.add_intersection_annotation(ax, x, y, formatter)

    # --------------- Add New Axis -------------#

    def save_axis(self):
        # Store the properties of existing axes
        old_axes_props = []
        for ax in self.axes:
            props = {
                'lines': [{'xdata': line.get_xdata(), 'ydata': line.get_ydata(),
                           'color': line.get_color(), 'label': line.get_label()} for line in ax.lines],
                'ylabel': ax.get_ylabel(),
                'title': ax.get_title(),
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
                'xscale': ax.get_xscale(),
                'yscale': ax.get_yscale()
            }
            if ax.get_legend():
                props['legend'] = ax.get_legend().texts
            old_axes_props.append(props)

        return old_axes_props

    def refresh_plots(self, old_axes_props, n):
        # Clear the figure and create new subplots
        self.fig.clear()
        self.axes = []
        for i in range(n):
            if i == 0:
                ax = self.fig.add_subplot(n, 1, i + 1)
            else:
                ax = self.fig.add_subplot(n, 1, i + 1, sharex=self.axes[0])

            # If this is an existing axis, restore its contents and properties
            if i < len(old_axes_props):
                props = old_axes_props[i]
                for line in props['lines']:
                    ax.plot(line['xdata'], line['ydata'],
                            color=line['color'], label=line['label'])
                ax.set_ylabel(props['ylabel'])
                ax.set_title(props['title'])
                ax.set_xlim(props['xlim'])
                ax.set_ylim(props['ylim'])
                ax.set_xscale(props['xscale'])
                ax.set_yscale(props['yscale'])
                if 'legend' in props:
                    # ax.legend(
                    #     [text.get_text() for text in props['legend']])
                    self.add_legend(ax, [text.get_text()
                                    for text in props['legend']])
                    # self.legend.set_picker(True)

                # ax.spines["right"].set_visible(False)
                # ax.spines["top"].set_visible(False)
            self.axes.append(ax)

        # Set the new axis as the current one
        self.current_ax = self.axes[-1]
        # self.blit_manager.on_draw(None)
        self.fig.canvas.flush_events()

        # Adjust the layout

        # plt.tight_layout(pad=4.0, h_pad=2.0, w_pad=2.0)

    def add_legend(self, ax, *args, **kwargs):
        legend = ax.legend(*args, **kwargs)
        legend.set_picker(True)
        self.legends[ax] = legend

        # Put legend beside plot
        # Set the initial position if not already set
        # if legend._loc == 0:  # 0 is the default value when no location is specified
        #     # Place it outside the plot on the upper right
        #     legend._loc = (1.01, 0)

        # Connect the pick event
        self.fig.canvas.mpl_connect('pick_event', self.on_legend_pick)

    def on_legend_pick(self, event):
        if event.artist in self.legends.values():
            legend = event.artist
            ax = legend.axes
            for legline, origline in zip(legend.get_lines(), ax.get_lines()):
                visible = not origline.get_visible()
                origline.set_visible(visible)
                legline.set_alpha(1.0 if visible else 0.2)
            self.fig.canvas.draw()

    def add_axis(self):
        # Calculate the new layout
        n = len(self.axes) + 1

        # Store the properties of existing axes
        old_axes_props = self.save_axis()

        # Refresh the plots with the new number of axes
        self.refresh_plots(old_axes_props, n)

        self.format_axis(self.axes)

    def remove_axis(self):
        # Check if there is more than one axis
        if len(self.axes) > 1:
            # Find the index of the current axis
            index_to_remove = self.axes.index(self.current_ax)

            # Store the properties of existing axes except the one to be removed
            old_axes_props = [props for i, props in enumerate(
                self.save_axis()) if i != index_to_remove]

            # Refresh the plots with the new number of axes
            self.refresh_plots(old_axes_props, len(self.axes) - 1)
            self.format_axis(self.axes)

            # Force a complete redraw of the figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Update the BlitManager
            self.blit_manager.on_draw(None)
            self.blit_manager.update()

            # Adjust the layout
            # self.fig.tight_layout()

            # Redraw the canvas again to ensure all changes are reflected
            self.fig.canvas.draw_idle()
        else:
            print("Cannot remove the only remaining axis")

    def setup_axes(self, ax):

        # for label in ax.get_yticklabels():
        #     label.set_ha("right")
        # for label in ax.get_xticklabels():
        #     pass
        #     label.set_va("bottom")
        #     # ax.set_ylabel("ha=left")
        return ax

    # --------------- Plotting -------------#

    def plot_settings(
        self,
        y: np.ndarray,
        x_limit: tuple = (),
        y_limit: tuple = (),
        x_scale: str = "",
        y_scale: str = "",
        x_eng_format: bool = True,
        y_eng_format: bool = True,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        save_fig: str = "",
    ):

        self.current_ax.set_title(title)
        self.current_ax.set_xlabel(x_label)
        self.current_ax.set_ylabel(y_label)
        self.setup_axes(self.current_ax)
        if x_limit:
            self.current_ax.set_xlim(*x_limit)

        if y_limit:
            self.current_ax.set_ylim(*y_limit)

        if x_scale:
            self.current_ax.set_xscale(x_scale)

        if y_scale:
            self.current_ax.set_yscale(y_scale)

        if y_eng_format:
            self.current_ax.yaxis.set_major_formatter(
                EngFormatter(unit="", sep=""))
        if x_eng_format:
            self.current_ax.xaxis.set_major_formatter(
                EngFormatter(unit="", sep=""))

        if save_fig:
            self.fig.savefig(save_fig, dpi=130)

    def quick_plot(
        self,
        x: np.ndarray | list | tuple,
        y: np.ndarray | list | tuple,
        *,
        x_label: str = "",
        y_label: str = "",
        x_limit: tuple = (),
        y_limit: tuple = (),
        x_scale: str = "",
        y_scale: str = "",
        x_eng_format: bool = True,
        y_eng_format: bool = True,
        legend: list = [],
        title: str = None,
        save_fig: str = "",
    ):
        self.plot_settings(
            y,
            x_limit,
            y_limit,
            x_scale,
            y_scale,
            x_eng_format,
            y_eng_format,
            x_label,
            y_label,
            title,
            save_fig,
        )

        self.__plot(x, y, legend, save_fig)
        self.autoSize()

    def __plot(self, x, y, legend, save_fig):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.ndim == y.ndim:
            self.current_ax.plot(x, y, picker=True)

        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            for x_, y_ in zip(x, y):
                if x_.ndim == 1 and x_.shape[0] != y_.shape[0]:
                    self.current_ax.plot(x_, y_.T, picker=True)
                else:
                    self.current_ax.plot(x_, y_, picker=True)

        elif x.ndim == 1:
            if x.shape[0] != y.shape[0]:
                self.current_ax.plot(x, y.T, picker=True)
            else:
                self.current_ax.plot(x, y, picker=True)

        elif y.ndim == 1:
            if y.shape[0] != x.shape[0]:
                self.current_ax.plot(x.T, y, picker=True)
            else:
                self.current_ax.plot(x, y, picker=True)

        if legend:
            # self.legend = self.current_ax.legend(
            #     legend)

            self.add_legend(self.current_ax, legend)

        if save_fig:
            self.fig.savefig(save_fig, bbox_inches="tight")

        self.blit_manager.update()

    def show_plot(self):
        plt.show()

    def save_plot(self, filename: str):
        self.fig.savefig(filename)

    def clear_plot(self, ax=None):
        """
        Clears the specified plot. If no axis is specified, clears the current axis.

        Parameters:
        ax (matplotlib.axes.Axes, optional): The axis to clear. If None, clears the current axis.
        """
        if ax is None:
            ax = self.current_ax
        ax.cla()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        # self.fig.canvas.draw()

    def set_scale(self, x_scale: str = "linear", y_scale: str = "linear"):
        self.current_ax.set_xscale(x_scale)
        self.current_ax.set_yscale(y_scale)

    def set_labels(self, x_label: str = "", y_label: str = ""):
        self.current_ax.set_xlabel(x_label)
        self.current_ax.set_ylabel(y_label)

    def set_title(self, title: str = ""):
        self.current_ax.set_title(title)

    def set_limits(self, x_limit: tuple = (), y_limit: tuple = ()):
        if x_limit:
            self.current_ax.set_xlim(*x_limit)
        if y_limit:
            self.current_ax.set_ylim(*y_limit)

    def autoSize(self):
        for ax in self.axes:
            if ax:
                ax.relim()
                ax.autoscale_view(True, True, True)
        self.blit_manager.update()
