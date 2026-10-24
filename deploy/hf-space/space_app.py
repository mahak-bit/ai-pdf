"""Hugging Face Space entry point (Gradio SDK, free CPU Basic hardware).

Serves the unchanged DocuMind FastAPI app on port 7860. A small Gradio page is
mounted at "/" so visitors to the Space see what it is; the API lives under /api
and interactive docs under /docs. API routes are registered before the mount,
so they take precedence.
"""

import gradio as gr
import uvicorn

from app.main import app

with gr.Blocks(title="DocuMind AI API") as status_page:
    gr.Markdown(
        "# DocuMind AI API\n"
        "Backend for DocuMind AI: retrieval-augmented question answering over PDFs, "
        "with answers cited to the page.\n\n"
        "- Health check: [`/api/health`](/api/health)\n"
        "- API docs: [`/docs`](/docs)\n"
        "- Source code: https://github.com/mahak-bit/ai-pdf"
    )

app = gr.mount_gradio_app(app, status_page, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
