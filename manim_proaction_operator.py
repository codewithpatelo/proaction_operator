from manim import *
import numpy as np


class ProActionOperatorScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#0a0a0f"
        self.camera.frame_width = 16
        self.camera.frame_height = 9

        neon_colors = ["#00ffcc", "#ff6b35", "#ff00ff", "#ff3366", "#00ccff", "#ffff00"]
        names = ["Attention", "Perception", "Hormonal", "Emotional", "Neuro-fast", "Cognitive"]
        short = ["A", "P", "H", "E", "N", "C"]
        setpoints = np.array([0.30, 0.20, 0.30, 0.10, 0.20, 0.40])
        lambdas = np.array([0.08, 0.07, 0.10, 0.08, 0.06, 0.10])
        alphas = np.array([0.20, 0.15, 0.25, 0.20, 0.10, 0.15])
        kappas = np.array([0.10, 0.10, 0.15, 0.12, 0.08, 0.10])
        W = np.array([
            [0.00, 0.05, 0.00, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00, 0.05, 0.00],
            [0.00, 0.00, 0.15, 0.00, 0.00, 0.05],
            [0.00, 0.00, 0.20, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00, 0.10, 0.15, 0.00],
        ])

        def neon_text(s, size=32, color="#ffffff"):
            return Text(s, font_size=size, color=color, weight=BOLD)

        def neon_eq(tex, size=38, color="#00ffcc"):
            return MathTex(tex, font_size=size, color=color)

        title = VGroup(
            neon_text("The Pro-Action Operator", 52, "#00ffcc"),
            neon_eq(r"\Gamma", 64, "#ff00ff")
        ).arrange(RIGHT, buff=0.3)
        subtitle = neon_text("Bio-inspired regulatory harness for LLM agents", 28, "#8888aa")
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(VGroup(title, subtitle), shift=UP), run_time=1.5)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, subtitle), shift=DOWN), run_time=0.8)

        eq = neon_eq(
            r"\mathbf{x}_{t+1} = \mathbf{x}_t - \boldsymbol{\kappa}\odot(\mathbf{x}_t-\mathbf{x}^{*}+\boldsymbol{\delta}_{\mathrm{pert}}) + \boldsymbol{\lambda} - \boldsymbol{\alpha}\odot g_t + W\mathbf{x}_t",
            34,
            "#ffffff"
        ).to_edge(UP, buff=0.8)
        self.play(Write(eq), run_time=2.0)
        self.wait(1.0)
        self.play(eq.animate.scale(0.85).to_edge(UP, buff=0.5), run_time=0.8)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            z_range=[0, 1, 0.2],
            x_length=6,
            y_length=6,
            z_length=6,
            color="#444466"
        )
        axes.set_z_index(0)
        self.play(FadeIn(axes), run_time=1.0)

        def simulate_trajectory(n_steps=50, perturb_round=10):
            x = np.array([0.8, 0.5, 0.9, 0.7, 0.6, 1.0])
            trajectory = [x.copy()]
            for t in range(n_steps):
                kappa_pert = np.zeros(6)
                if t == perturb_round:
                    kappa_pert[2] = 0.5
                delta = x - setpoints + kappa_pert
                g = 0.6
                x_next = x - kappas * delta + lambdas - alphas * g + W @ x
                x_next = np.clip(x_next, 0.0, 1.0)
                x = x_next
                trajectory.append(x.copy())
            return np.array(trajectory)

        traj = simulate_trajectory()
        traj_3d = traj[:, [2, 3, 5]]

        lines = VGroup()
        for i in range(len(traj_3d) - 1):
            line = Line3D(
                start=traj_3d[i],
                end=traj_3d[i + 1],
                color=neon_colors[2],
                stroke_width=3
            )
            lines.add(line)

        self.play(Create(lines), run_time=2.5)
        self.wait(0.5)

        start_dot = Dot3D(point=traj_3d[0], color="#00ffcc", radius=0.08)
        end_dot = Dot3D(point=traj_3d[-1], color="#ff00ff", radius=0.08)
        perturb_dot = Dot3D(point=traj_3d[10], color="#ff3366", radius=0.12)
        self.play(FadeIn(start_dot), FadeIn(end_dot), FadeIn(perturb_dot), run_time=0.8)

        axis_labels = VGroup(
            neon_text("Hormonal", 20, neon_colors[2]).next_to(axes.x_axis.get_end(), RIGHT),
            neon_text("Emotional", 20, neon_colors[3]).next_to(axes.y_axis.get_end(), UP),
            neon_text("Cognitive", 20, neon_colors[5]).next_to(axes.z_axis.get_end(), OUT),
        )
        self.play(FadeIn(axis_labels), run_time=0.6)

        perturb_label = neon_text("Perturbation at t=10", 24, "#ff3366")
        perturb_label.next_to(perturb_dot, UP)
        self.play(FadeIn(perturb_label), run_time=0.5)
        self.wait(1.5)

        self.play(FadeOut(VGroup(axes, lines, start_dot, end_dot, perturb_dot, axis_labels, perturb_label)), run_time=1.0)
        self.move_camera(phi=0 * DEGREES, theta=0 * DEGREES)
        self.wait(0.5)

        subsystems = VGroup()
        for i, (name, color) in enumerate(zip(names, neon_colors)):
            row = VGroup(
                neon_text(f"{short[i]}: {name}", 28, color),
                neon_text(f"λ={lambdas[i]:.2f} κ={kappas[i]:.2f} x*={setpoints[i]:.2f}", 22, "#8888aa")
            ).arrange(RIGHT, buff=0.3)
            subsystems.add(row)
        subsystems.arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(LEFT * 2 + UP * 0.5)
        self.play(FadeIn(subsystems, lag_ratio=0.1), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(subsystems), run_time=0.8)

        coupling_graph = VGroup()
        positions = [
            LEFT * 3 + UP * 2,
            LEFT * 1 + UP * 2,
            RIGHT * 1 + UP * 2,
            RIGHT * 3 + UP * 0,
            RIGHT * 1 + DOWN * 2,
            LEFT * 1 + DOWN * 2,
        ]
        nodes = VGroup()
        for pos, color in zip(positions, neon_colors):
            circle = Circle(radius=0.35, color=color, stroke_width=3, fill_opacity=0.3)
            circle.move_to(pos)
            nodes.add(circle)
        coupling_graph.add(nodes)

        edges = [(0, 1, 0.05), (1, 0, 0.05), (2, 1, 0.10), (2, 4, 0.05),
                 (3, 2, 0.15), (3, 5, 0.05), (4, 2, 0.20), (5, 0, 0.05),
                 (5, 3, 0.10), (5, 4, 0.15)]
        arrows = VGroup()
        for target, source, weight in edges:
            start = positions[source]
            end = positions[target]
            arrow = Arrow(
                start + (end - start) * 0.15,
                end - (end - start) * 0.15,
                color="#ffffff",
                stroke_width=2 + weight * 30,
                max_tip_length_to_length_ratio=0.2
            )
            arrows.add(arrow)
        coupling_graph.add(arrows)

        w_title = neon_text("Sparse Coupling Matrix W", 36, "#ff00ff")
        w_title.to_edge(UP, buff=0.6)
        self.play(FadeIn(w_title), FadeIn(coupling_graph, lag_ratio=0.1), run_time=1.8)
        self.wait(1.5)
        self.play(FadeOut(VGroup(w_title, coupling_graph)), run_time=0.8)

        closing = VGroup(
            neon_text("Γ makes activation experimentally addressable", 42, "#00ffcc"),
            neon_text("stateful · inspectable · perturbable", 32, "#8888aa"),
            neon_text("future: matched controls, longer horizons, human designs", 26, "#666688"),
        ).arrange(DOWN, buff=0.4)
        self.play(FadeIn(closing, shift=UP), run_time=1.2)
        self.wait(2.0)
        self.play(FadeOut(closing), run_time=1.0)
