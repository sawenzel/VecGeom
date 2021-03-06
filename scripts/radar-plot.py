#!/usr/bin/env python

"""
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        # def setlinestyle(self, ls):

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path, linewidth=2)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


if __name__ == '__main__':
    # Read data file
    with open(sys.argv[1]) as f:
        caption = f.readline().strip()

        line = f.readline()
        N, M = line.split(",")
        N = int(N)
        M = int(M)

        line = f.readline().strip()
        columns = line.split(",")
        assert len(columns) == N
        data = {}
        data["column names"] = columns
        data[caption] = []

        line = f.readline().strip()
        factors = line.split(",")
        assert len(factors) == M

        # For each factor
        for m in range(0,M):
            line = f.readline().strip()
            line = line.split(",")
            line = [float(x) for x in line]
            print(line)
            assert len(line) == N
            data[caption].append(line)

        data[caption].append( [0] * N)

        print("Got N = {0}, M = {1}".format(N, M))
        print(columns)
        # sys.exit(1)
    
    # N = N+1
    mpl.rcParams['axes.linewidth'] = 2
    # mpl.rcParams['lines.linewidth'] = 2
    theta = radar_factory(N, frame='polygon')

    # data = get_data()
    spoke_labels = data.pop('column names')

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['#C7F464', '#C44D58', '#4ECDC4', '#556270', 'c', 'k']
    while len(colors) > len(factors)+1:
        colors.pop()
    print(colors)
    # Plot the four cases from the example data on separate axes
    for n, title in enumerate(data.keys()):
        ax = fig.add_subplot(1, 1, n+1, projection='radar')
        # ax = fig.add_subplot(111, projection='radar')
        plt.rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(data[title], colors):
            ax.plot(theta, d, color=color, linestyle="None")
            p = ax.fill(theta, d, facecolor=color, linestyle="solid", linewidth="2", edgecolor="#000000")
            p = p[0]
            print(type(p))
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    plt.subplot(1, 1, 1)
    labels = tuple(factors)

    # labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1, shadow=False)
    plt.setp(legend.get_texts(), fontsize='small')

    # plt.figtext(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
    #             ha='center', color='black', weight='bold', size='large')
    plt.show()
