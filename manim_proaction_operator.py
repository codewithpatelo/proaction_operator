from manim import *
import numpy as np


class ProActionOperatorScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#0a0a0f"

        title = Text("The Pro-Action Operator Γ", font_size=56, color="#00ffcc", weight=BOLD)
        self.play(FadeIn(title), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(title), run_time=0.5)

        eq = MathTex(
            r"\mathbf{x}_{t+1} = \mathbf{x}_t - \boldsymbol{\kappa}(\mathbf{x}_t-\mathbf{x}^{*}) + \boldsymbol{\lambda} - \boldsymbol{\alpha}g_t + W\mathbf{x}_t",
            font_size=42,
            color="#ffffff"
        )
        self.play(Write(eq), run_time=2.0)
        self.wait(1.5)
        self.play(eq.animate.scale(0.7).to_edge(UP, buff=0.5), run_time=0.5)

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        axes = ThreeDAxes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            z_range=[0, 1, 0.2],
            x_length=6,
            y_length=6,
            z_length=6,
            color="#666688"
        )
        self.play(FadeIn(axes), run_time=1.0)

        def gamma_surface(x, y):
            x_star = 0.3
            kappa = 0.1
            lam = 0.08
            z = x - kappa * (x - x_star) + lam
            return np.clip(z, 0, 1)

        resolution = 20
        x_vals = np.linspace(0, 1, resolution)
        y_vals = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = gamma_surface(X, Y)

        surface = Surface(
            lambda u, v: np.array([u, v, gamma_surface(u, v)]),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(resolution, resolution),
            fill_color="#00ffcc",
            fill_opacity=0.3,
            stroke_color="#00ffcc",
            stroke_width=1
        )
        self.play(Create(surface), run_time=2.0)
        self.wait(1.5)

        self.play(FadeOut(VGroup(axes, surface, eq)), run_time=1.0)
