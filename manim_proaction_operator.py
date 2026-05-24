from manim import *
import numpy as np

class ProActionOperator(ThreeDScene):
    def construct(self):
        # 1. Background setup
        self.camera.background_color = "#0a0a0f"

        # 2. Title
        title = Text("The Pro-Action Operator Γ", font_size=48, weight=BOLD, color="#00ffcc")
        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeOut(title))

        # 3. Equation (MathTex for exact mathematical typography)
        equation = MathTex(
            r"x_{t+1} = x_t - \kappa(x_t - x^*) + \lambda - \alpha \cdot g_t + W \cdot x_t",
            font_size=42,
            color="#ffffff"
        )
        self.play(Write(equation), run_time=2)
        self.wait(1.5)
        
        # Scale and move to top
        self.play(equation.animate.scale(0.7).to_edge(UP))

        # 4. 3D Visualization setup
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[0, 2, 0.5],
            x_length=5,
            y_length=5,
            z_length=4
        )

        # EXACT MATHEMATICAL DYNAMICS
        # Parameters mapped directly from your Gamma operator
        kappa = 0.8    # Homeostatic anchor strength
        lam = 0.2      # Drift/Bias (scalar representation)
        alpha = 0.5    # Active inference/learning rate
        W_val = 0.3    # Systemic momentum weight

        def gamma_dynamics_surface(u, v):
            # State vector x_t
            x = u
            y = v
            
            # g_t: local gradient of an arbitrary loss function (e.g., sin(3x)*cos(3y))
            g_x = 3 * np.cos(3*x) * np.cos(3*y)
            g_y = -3 * np.sin(3*x) * np.sin(3*y)
            
            # Gamma update equation terms evaluated exactly
            # delta = -k(x - x*) + lambda - alpha(g_t) + W(x_t)
            delta_x = -kappa * (x - 0) + lam - alpha * g_x + W_val * x
            delta_y = -kappa * (y - 0) + lam - alpha * g_y + W_val * y
            
            # Z represents the magnitude (norm) of the Gamma operator's response at state (x,y)
            z = np.sqrt(delta_x**2 + delta_y**2)
            
            # Limit Z strictly from 0 to 1 as requested in original prompt
            z_scaled = np.clip(z * 0.5, 0, 1) 
            
            return axes.c2p(x, y, z_scaled)

        surface = Surface(
            gamma_dynamics_surface,
            u_range=[-1, 1],
            v_range=[-1, 1],
            resolution=(20, 20),
            fill_color="#00ffcc",
            fill_opacity=0.3,
            stroke_color="#00ffcc",
            stroke_width=1
        )

        # Plotting
        self.play(Create(axes), run_time=1)
        self.play(Create(surface), run_time=2)
        self.wait(1.5)

        # 5. Fade out everything
        self.play(FadeOut(Group(axes, surface, equation)))
        self.wait(1)
