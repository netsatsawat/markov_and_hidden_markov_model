"""Draw small Markov chains as state diagrams.

The layout is deliberate rather than automatic: for the 2, 3 and 4 state
chains a tutorial needs, a fixed arrangement beats a force-directed one
every time. Arrows curve so that a pair of opposite transitions (A to B
and B to A) reads as two distinct arrows instead of one line with two
heads, and every probability sits in a small white box on its own arrow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MarkovChain:

    def __init__(self, M, labels, node_colors=None, node_centers=None):
        """
        Draw a Markov chain from its transition matrix.

        Inputs:
            - M            transition matrix (rows must sum to 1)
            - labels       one label per state
            - node_colors  optional dict {label: hex color} or list of colors;
                           states not listed fall back to the default blue
            - node_centers optional list of (x, y) centers, one per state,
                           for custom layouts with any number of states
        """
        M = np.asarray(M, dtype=float)
        if M.shape[0] < 2:
            raise Exception("There should be at least 2 states")
        if M.shape[0] != M.shape[1]:
            raise Exception("Transition matrix should be square")
        if M.shape[0] != len(labels):
            raise Exception("There should be as many labels as states")
        if M.shape[0] > 4 and node_centers is None:
            raise Exception("Pass node_centers for chains with more than 4 states")

        self.M = M
        self.n_states = M.shape[0]
        self.labels = list(labels)

        self.default_color = '#2a78d6'
        self.arrow_color = '#9097a1'
        self.label_fontsize = 11
        self.node_radius = 0.5

        if node_colors is None:
            self.node_colors = [self.default_color] * self.n_states
        elif isinstance(node_colors, dict):
            self.node_colors = [node_colors.get(l, self.default_color)
                                for l in self.labels]
        else:
            self.node_colors = list(node_colors)

        self.set_node_centers(node_centers)

    def set_node_centers(self, node_centers=None):
        """Fixed layouts for 2 to 4 states, or a caller-supplied one."""
        if node_centers is not None:
            self.node_centers = [np.asarray(c, dtype=float) for c in node_centers]
        elif self.n_states == 2:
            self.node_centers = [np.array([-2.2, 0.0]), np.array([2.2, 0.0])]
        elif self.n_states == 3:
            self.node_centers = [np.array([-2.4, -1.6]), np.array([2.4, -1.6]),
                                 np.array([0.0, 2.0])]
        else:
            self.node_centers = [np.array([-2.4, 2.4]), np.array([2.4, 2.4]),
                                 np.array([2.4, -2.4]), np.array([-2.4, -2.4])]

        xs = [c[0] for c in self.node_centers]
        ys = [c[1] for c in self.node_centers]
        pad = 2.0
        self.xlim = (min(xs) - pad, max(xs) + pad)
        self.ylim = (min(ys) - pad, max(ys) + pad)
        w = self.xlim[1] - self.xlim[0]
        h = self.ylim[1] - self.ylim[0]
        scale = 8.5 / max(w, h)
        self.figsize = (w * scale, h * scale)

    def _add_node(self, ax, center, label, color):
        circle = mpatches.Circle(center, self.node_radius, facecolor=color,
                                 edgecolor='white', linewidth=2.0, zorder=3)
        ax.add_patch(circle)
        ax.annotate(label, xy=center, color='white', ha='center', va='center',
                    fontsize=13, fontweight='bold', zorder=4)

    def _add_arrow(self, ax, i, j, prob, curved):
        """Curved arrow from state i to state j with the probability on it."""
        p0, p1 = self.node_centers[i], self.node_centers[j]
        d = p1 - p0
        dist = np.hypot(*d)
        u = d / dist
        r = self.node_radius

        start = p0 + u * (r + 0.05)
        end = p1 - u * (r + 0.12)
        rad = 0.22 if curved else 0.08

        arrow = mpatches.FancyArrowPatch(
            start, end, connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', mutation_scale=17,
            linewidth=1.6, color=self.arrow_color, zorder=1)
        ax.add_patch(arrow)

        # arc3 places its bezier control point to the right of the travel
        # direction for positive rad, so the label has to follow that side.
        # Labels sit at the curve midpoint for paired arrows and closer to
        # the origin for one-way arrows, so crossing diagonals stay legible.
        chord = (start + end) / 2
        side = np.array([u[1], -u[0]])
        control = chord + side * (rad * dist)
        t = 0.5 if curved else 0.38
        pos = ((1 - t) ** 2 * start + 2 * (1 - t) * t * control
               + t ** 2 * end)
        ax.annotate(f'{prob:g}', xy=pos, ha='center', va='center',
                    fontsize=self.label_fontsize, color='#333333', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='#d5d5d5', linewidth=0.8))

    def _add_self_loop(self, ax, i, prob):
        """A loop on the side of the node that faces away from the rest."""
        center = self.node_centers[i]
        centroid = np.mean(self.node_centers, axis=0)
        out = center - centroid
        norm = np.hypot(*out)
        out = np.array([0.0, 1.0]) if norm < 1e-9 else out / norm
        theta = np.arctan2(out[1], out[0])
        r = self.node_radius

        a_ang, b_ang = theta + 0.65, theta - 0.65
        a = center + r * np.array([np.cos(a_ang), np.sin(a_ang)])
        b = center + r * np.array([np.cos(b_ang), np.sin(b_ang)])

        # Negative rad so the loop bulges away from the node instead of
        # hiding behind it (arc3 bows right of the travel direction).
        arrow = mpatches.FancyArrowPatch(
            a, b, connectionstyle='arc3,rad=-2.4',
            arrowstyle='-|>', mutation_scale=15,
            linewidth=1.6, color=self.arrow_color, zorder=1)
        ax.add_patch(arrow)

        label_pos = center + out * (r + 0.95)
        ax.annotate(f'{prob:g}', xy=label_pos, ha='center', va='center',
                    fontsize=self.label_fontsize, color='#333333', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='#d5d5d5', linewidth=0.8))

    def draw(self, img_path=None, title=None):
        """Draw the chain; optionally save it to img_path."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal')
        ax.axis('off')

        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    if self.M[i, j] > 0:
                        self._add_self_loop(ax, i, self.M[i, j])
                elif self.M[i, j] > 0:
                    self._add_arrow(ax, i, j, self.M[i, j],
                                    curved=self.M[j, i] > 0)

        for center, label, color in zip(self.node_centers, self.labels,
                                        self.node_colors):
            self._add_node(ax, center, label, color)

        if title:
            ax.set_title(title, fontsize=13, fontweight='semibold',
                         color='#333333', pad=14)

        if img_path:
            fig.savefig(img_path, dpi=150, bbox_inches='tight',
                        facecolor='white')
        plt.show()
