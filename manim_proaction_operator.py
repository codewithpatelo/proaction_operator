from manim import *
import numpy as np


class ProActionOperatorScene(MovingCameraScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.camera.frame_width = 14
        self.camera.frame_height = 8

        colors = [BLUE_E, ORANGE, GREEN_E, RED_E, PURPLE_E, "#8B4513"]
        names = ["Attention", "Perception", "Hormonal/arousal", "Emotional", "Readiness", "Cognitive"]
        short = ["A", "P", "H", "E", "N", "C"]
        setpoints = [0.30, 0.20, 0.30, 0.10, 0.20, 0.40]
        lambdas = [0.08, 0.07, 0.10, 0.08, 0.06, 0.10]
        values_before = [0.30, 0.20, 0.30, 0.10, 0.20, 0.40]
        values_after = [0.38, 0.29, 0.72, 0.24, 0.35, 0.48]
        values_recovered = [0.32, 0.23, 0.36, 0.13, 0.23, 0.42]

        def text(s, size=30, color=BLACK, weight=NORMAL):
            return Text(s, font_size=size, color=color, weight=weight)

        def label_box(label, width=2.2, height=0.75, color=BLACK, fill=WHITE):
            box = RoundedRectangle(corner_radius=0.12, width=width, height=height, color=color, fill_color=fill, fill_opacity=1, stroke_width=2)
            lab = text(label, 24, color)
            lab.move_to(box)
            return VGroup(box, lab)

        def thermostat_row(name, symbol, color, setpoint, value):
            symbol_tex = MathTex(rf"x_{{{symbol}}}", font_size=36, color=color)
            name_tex = text(name, 20, BLACK)
            left = VGroup(symbol_tex, name_tex).arrange(DOWN, buff=0.05).set_width(1.35)
            bar = Line(LEFT * 1.6, RIGHT * 1.6, color=BLACK, stroke_width=5)
            bar.next_to(left, RIGHT, buff=0.35)
            marker = Line(UP * 0.18, DOWN * 0.18, color=DARK_GRAY, stroke_width=4)
            marker.move_to(bar.get_left() + RIGHT * 3.2 * setpoint)
            dot = Dot(color=color, radius=0.09)
            dot.move_to(bar.get_left() + RIGHT * 3.2 * value)
            low = text("0", 14, DARK_GRAY).next_to(bar.get_left(), DOWN, buff=0.08)
            high = text("1", 14, DARK_GRAY).next_to(bar.get_right(), DOWN, buff=0.08)
            return VGroup(left, bar, marker, dot, low, high)

        title = MathTex(r"\text{The Pro-Action Operator }\Gamma", font_size=58, color=BLACK)
        subtitle = text("A coupled regulatory harness around LLM policy execution", 28, DARK_GRAY)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(VGroup(title, subtitle), shift=DOWN), run_time=1.2)
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, subtitle), shift=UP), run_time=0.6)

        heading = text("Six-dimensional regulatory state", 32, BLACK, BOLD).to_edge(UP, buff=0.55)
        rows = VGroup(*[
            thermostat_row(names[i], short[i], colors[i], setpoints[i], values_before[i])
            for i in range(6)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).scale(0.92).shift(LEFT * 2.8 + DOWN * 0.15)
        vector = Matrix([[rf"x_{{{s}}}(t)"] for s in short], element_to_mobject_config={"font_size": 26, "color": BLACK})
        vector.next_to(rows, RIGHT, buff=1.0)
        vector_label = MathTex(r"\mathbf{x}_t", font_size=40, color=BLACK).next_to(vector, UP, buff=0.2)
        self.play(FadeIn(heading), FadeIn(rows, lag_ratio=0.06), FadeIn(vector), FadeIn(vector_label), run_time=1.5)
        self.wait(0.8)
        self.play(*[
            rows[i][3].animate.move_to(rows[i][1].get_left() + RIGHT * 3.2 * values_after[i])
            for i in range(6)
        ], run_time=1.2)
        pulse_note = text("Round-10 perturbation loads arousal", 24, RED_E).next_to(rows[2], RIGHT, buff=0.6)
        self.play(FadeIn(pulse_note, shift=LEFT), rows[2][3].animate.scale(1.45), run_time=0.6)
        self.play(rows[2][3].animate.scale(1 / 1.45), run_time=0.4)
        self.wait(0.7)
        self.play(FadeOut(VGroup(heading, rows, vector, vector_label, pulse_note)), run_time=0.7)

        eq = MathTex(
            r"\mathbf{x}_{t+1}", "=", r"\mathbf{x}_t", "-", r"\boldsymbol{\kappa}\odot(\mathbf{x}_t-\mathbf{x}^{*}+\boldsymbol{\delta}_{\mathrm{pert}}(t))", "+", r"\boldsymbol{\lambda}", "-", r"\boldsymbol{\alpha}\odot g_t", "+", r"W\mathbf{x}_t",
            font_size=31,
            color=BLACK,
        ).to_edge(UP, buff=0.75)
        self.play(Write(eq), run_time=1.4)
        meanings = [
            (2, "current regulatory state", BLUE_E),
            (4, "return toward set-points plus transient load", GREEN_E),
            (6, "baseline drift", ORANGE),
            (8, "action-quality feedback", RED_E),
            (10, "sparse subsystem coupling", PURPLE_E),
        ]
        cards = VGroup()
        for idx, label, color in meanings:
            card = label_box(label, width=4.6, height=0.55, color=color).scale(0.82)
            cards.add(card)
        cards.arrange(DOWN, aligned_edge=LEFT, buff=0.22).shift(DOWN * 0.75)
        for k, (idx, _, color) in enumerate(meanings):
            self.play(eq[idx].animate.set_color(color), FadeIn(cards[k], shift=RIGHT), run_time=0.55)
            self.wait(0.25)
            self.play(eq[idx].animate.set_color(BLACK), run_time=0.25)
        self.wait(0.8)
        self.play(FadeOut(VGroup(eq, cards)), run_time=0.7)

        graph_title = text("Sparse linear coupling W", 32, BLACK, BOLD).to_edge(UP, buff=0.55)
        node_positions = [
            LEFT * 3.0 + UP * 1.4,
            LEFT * 1.0 + UP * 1.4,
            RIGHT * 1.0 + UP * 1.4,
            RIGHT * 3.0 + UP * 0.0,
            RIGHT * 1.0 + DOWN * 1.4,
            LEFT * 1.0 + DOWN * 1.4,
        ]
        nodes = VGroup()
        for pos, s, name, color in zip(node_positions, short, names, colors):
            circle = Circle(radius=0.34, color=color, fill_color=color, fill_opacity=0.22, stroke_width=4).move_to(pos)
            sym = MathTex(rf"x_{{{s}}}", font_size=30, color=BLACK).move_to(circle)
            lab = text(name, 16, BLACK).next_to(circle, DOWN, buff=0.12)
            nodes.add(VGroup(circle, sym, lab))
        pair_weights = [(0, 1, 0.05), (1, 0, 0.05), (2, 1, 0.10), (2, 4, 0.05), (3, 2, 0.15), (3, 5, 0.05), (4, 2, 0.20), (5, 0, 0.05), (5, 3, 0.10), (5, 4, 0.15)]
        arrows = VGroup()
        for target, source, weight in pair_weights:
            start = nodes[source][0].get_center()
            end = nodes[target][0].get_center()
            direction = end - start
            unit = direction / np.linalg.norm(direction)
            arrow = Arrow(start + unit * 0.42, end - unit * 0.42, buff=0, color=DARK_GRAY, stroke_width=2 + 16 * weight, max_tip_length_to_length_ratio=0.16)
            arrows.add(arrow)
        w_caption = text("Edges visualize non-zero entries in the executed W matrix", 22, DARK_GRAY).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(graph_title), FadeIn(nodes, lag_ratio=0.08), Create(arrows, lag_ratio=0.08), FadeIn(w_caption), run_time=1.8)
        self.wait(1.2)
        self.play(FadeOut(VGroup(graph_title, nodes, arrows, w_caption)), run_time=0.7)

        loop_title = text("Benchmark control loop", 32, BLACK, BOLD).to_edge(UP, buff=0.55)
        history = label_box("IPD history\n+ opponent", 2.1, 0.95, BLUE_E)
        gamma = label_box("Γ update", 1.7, 0.75, GREEN_E)
        rsvi = label_box("RSVI\nprompt state", 2.0, 0.95, PURPLE_E)
        llm = label_box("LLM policy\nexecutor", 2.0, 0.95, BLACK)
        action = label_box("C / D action", 1.8, 0.75, ORANGE)
        quality = label_box("quality gₜ", 1.8, 0.75, RED_E)
        loop_nodes = VGroup(history, gamma, rsvi, llm, action, quality).arrange(RIGHT, buff=0.45).scale(0.85).shift(UP * 0.3)
        loop_arrows = VGroup(*[
            Arrow(loop_nodes[i].get_right(), loop_nodes[i + 1].get_left(), buff=0.08, color=BLACK, stroke_width=3)
            for i in range(5)
        ])
        feedback = CurvedArrow(quality.get_bottom(), gamma.get_bottom(), angle=-TAU / 4, color=RED_E, stroke_width=3)
        note = text("Γ modulates; the LLM still chooses the action", 25, DARK_GRAY).next_to(loop_nodes, DOWN, buff=0.75)
        self.play(FadeIn(loop_title), FadeIn(loop_nodes, lag_ratio=0.08), Create(loop_arrows), Create(feedback), FadeIn(note), run_time=1.8)
        for obj in loop_nodes:
            self.play(Indicate(obj, color=YELLOW_E, scale_factor=1.04), run_time=0.35)
        self.wait(0.7)
        self.play(FadeOut(VGroup(loop_title, loop_nodes, loop_arrows, feedback, note)), run_time=0.7)

        perturb_title = text("Perturbation and recovery", 32, BLACK, BOLD).to_edge(UP, buff=0.55)
        time_axis = NumberLine(x_range=[1, 50, 9], length=10, include_numbers=True, color=BLACK).shift(DOWN * 2.45)
        marker = Triangle(color=RED_E, fill_color=RED_E, fill_opacity=1).scale(0.16).rotate(PI).next_to(time_axis.n2p(10), UP, buff=0.1)
        marker_label = text("round 10", 18, RED_E).next_to(marker, UP, buff=0.08)
        recovery_rows = VGroup(*[
            thermostat_row(names[i], short[i], colors[i], setpoints[i], values_before[i])
            for i in range(6)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.15).scale(0.72).shift(LEFT * 2.5 + UP * 0.45)
        self.play(FadeIn(perturb_title), FadeIn(time_axis), FadeIn(recovery_rows, lag_ratio=0.05), FadeIn(marker), FadeIn(marker_label), run_time=1.2)
        self.play(*[
            recovery_rows[i][3].animate.move_to(recovery_rows[i][1].get_left() + RIGHT * 3.2 * values_after[i])
            for i in range(6)
        ], run_time=0.8)
        self.play(*[
            recovery_rows[i][3].animate.move_to(recovery_rows[i][1].get_left() + RIGHT * 3.2 * values_recovered[i])
            for i in range(6)
        ], run_time=1.4)
        self.wait(0.7)
        self.play(FadeOut(VGroup(perturb_title, time_axis, recovery_rows, marker, marker_label)), run_time=0.7)

        result_title = text("Opponent-differentiated behavior", 32, BLACK, BOLD).to_edge(UP, buff=0.55)
        opponents = ["TFT", "GTFT", "Grim", "Random"]
        scores = [0.827, 0.839, 0.495, 0.719]
        bars = VGroup()
        for opponent, score, color in zip(opponents, scores, [BLUE_E, GREEN_E, RED_E, ORANGE]):
            lab = text(opponent, 25, BLACK).set_width(1.2)
            bar = Rectangle(width=4.1 * score, height=0.38, color=color, fill_color=color, fill_opacity=0.55, stroke_width=1.5)
            value = text(f"{score:.3f}", 20, BLACK)
            row = VGroup(lab, bar, value).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)
            bars.add(row)
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.32).shift(LEFT * 1.9 + UP * 0.25)
        result_note = text("Main claim: inspectable regulatory state changes the behavioral profile, not that Γ maximizes IPD payoff", 21, DARK_GRAY).to_edge(DOWN, buff=0.72)
        self.play(FadeIn(result_title), FadeIn(bars, lag_ratio=0.12), FadeIn(result_note), run_time=1.5)
        self.wait(1.2)
        self.play(FadeOut(VGroup(result_title, bars, result_note)), run_time=0.7)

        closing = VGroup(
            text("Γ makes activation experimentally addressable", 34, BLACK, BOLD),
            text("stateful · inspectable · perturbable", 28, DARK_GRAY),
            text("future work: matched controls, longer horizons, human-comparison designs", 22, DARK_GRAY),
        ).arrange(DOWN, buff=0.32)
        self.play(FadeIn(closing, shift=UP), run_time=1.0)
        self.wait(1.8)
        self.play(FadeOut(closing), run_time=0.8)
