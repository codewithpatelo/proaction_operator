from manim import *
import numpy as np

class ProActionOperator(ThreeDScene):
    def construct(self):
        # Configuración del fondo Dark Theme
        self.camera.background_color = "#0a0a0f"

        # 1. Título "The Pro-Action Operator Γ"
        title = Text("The Pro-Action Operator Γ", font_size=48, weight=BOLD, color="#00ffcc")
        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeOut(title))

        # 2. Ecuación de actualización Gamma
        # Usamos MathTex para fidelidad matemática impecable
        equation = MathTex(
            r"x_{t+1} = x_t - \kappa(x_t - x^*) + \lambda - \alpha \cdot g_t + W \cdot x_t",
            font_size=42,
            color=WHITE
        )
        self.play(Write(equation), run_time=2)
        self.wait(1.5)
        
        # Escalar y mover hacia arriba para dar espacio al plano 3D
        self.play(equation.animate.scale(0.7).to_edge(UP))

        # 3. Visualización 3D
        # Configurar ángulo de cámara: phi=60°, theta=-45°
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Ejes 3D (ajustados estéticamente alrededor del origen)
        axes = ThreeDAxes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            z_range=[-1, 3, 1],
            x_length=5,
            y_length=5,
            z_length=4
        ).set_color(WHITE).set_opacity(0.5)

        # Superficie Matemática (Fidelidad de la dinámica)
        # Representa la tensión: z = k*(x^2 + y^2) [Homeostasis] + a*cos(3x)*sin(3y) [Gradiente activo]
        def gamma_dynamics_surface(u, v):
            x = u
            y = v
            # El término cuadrático tira hacia el centro (kappa), las ondas representan la búsqueda de mínimos locales (alpha * g_t)
            z = 0.4 * (u**2 + v**2) + 0.3 * np.cos(3*u) * np.sin(3*v)
            return axes.c2p(x, y, z)

        surface = Surface(
            gamma_dynamics_surface,
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(20, 20),
            fill_color="#00ffcc",
            fill_opacity=0.3,
            stroke_color="#00ffcc",
            stroke_width=1
        )

        # Animación de la gráfica
        self.play(Create(axes))
        
        # Animación sutil de la cámara girando levemente para el "eye candy" 3D
        self.begin_ambient_camera_rotation(rate=0.1)
        
        self.play(Create(surface), run_time=2)
        self.wait(1.5)
        
        self.stop_ambient_camera_rotation()

        # 4. Fade out everything
        self.play(FadeOut(Group(axes, surface, equation)))
        self.wait(1)
