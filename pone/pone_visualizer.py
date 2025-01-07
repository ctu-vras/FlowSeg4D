import gradio as gr
import os


def load_mesh(mesh_file_name):
    return mesh_file_name


demo = gr.Interface(
    
    fn=load_mesh,
    inputs=gr.Model3D(height=920, label="3D Model", clear_color=[1.0, 1.0, 1.0, 1.0]),
    outputs=[],
    examples=[
        [os.path.join(os.path.dirname(__file__), "output_prediction.obj")],
        [os.path.join(os.path.dirname(__file__), "output_cluster.obj")],
        [os.path.join(os.path.dirname(__file__), "output_label.obj")],
    ],
)

if __name__ == "__main__":
    demo.launch()