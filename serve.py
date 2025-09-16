# dualcellquant/serve.py
from fastapi import FastAPI
import gradio as gr
from dualCellQuant import build_ui  # ← 既存の build_ui() を利用

demo = build_ui().queue()
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/dualcellquant") 
