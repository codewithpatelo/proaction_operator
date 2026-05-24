from manim import *
import numpy as np


class ProActionOperatorScene(Scene):
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
        self.play(FadeOut(eq), run_time=0.5)

        axes = Axes(
            x_range=[0, 50, 10],
            y_range=[0, 1, 0.2],
            x_length=10,
            y_length=5,
            color="#666688"
        )
        axes_labels = axes.get_axis_labels(x_label="Round t", y_label="State x")
        self.play(FadeIn(axes), FadeIn(axes_labels), run_time=1.0)

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
        colors = ["#00ffcc", "#ff6b35", "#ff00ff", "#ff3366", "#00ccff", "#ffff00"]
        names = ["Attention", "Perception", "Hormonal", "Emotional", "Neuro-fast", "Cognitive"]

        graphs = VGroup()
        for i in range(6):
            graph = axes.plot_line_graph(
                x_values=range(len(traj)),
                y_values=traj[:, i],
                line_color=colors[i],
                stroke_width=3
            )
            graphs.add(graph)

        self.play(Create(graphs), run_time=2.0)
        self.wait(1.0)

        legend = VGroup()
        for i, (name, color) in enumerate(zip(names, colors)):
            item = VGroup(
                Square(side_length=0.15, color=color, fill_opacity=1),
                Text(name, font_size=20, color="#ffffff")
            ).arrange(RIGHT, buff=0.2)
            legend.add(item)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(RIGHT, buff=0.5)
        self.play(FadeIn(legend), run_time=0.8)
        self.wait(1.5)

        perturb_line = DashedLine(
            axes.c2p(10, 0),
            axes.c2p(10, 1),
            color="#ff3366",
            stroke_width=2
        )
        perturb_label = Text("Perturbation", font_size=18, color="#ff3366")
        perturb_label.next_to(perturb_line, UP)
        self.play(Create(perturb_line), FadeIn(perturb_label), run_time=0.8)
        self.wait(1.5)

        self.play(FadeOut(VGroup(axes, axes_labels, graphs, legend, perturb_line, perturb_label)), run_time=1.0)

        closing = Text("Γ: stateful · inspectable · perturbable", font_size=40, color="#00ffcc", weight=BOLD)
        self.play(FadeIn(closing), run_time=1.0)
        self.wait(2.0)
        self.play(FadeOut(closing), run_time=0.8)
