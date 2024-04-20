import gradio as gr

def create_UI():
    with gr.Blocks(title= "RAG Chatbot",
        theme = "Soft"
        ) as demo:
        with gr.Column():
            with gr.Row():
                chat_history = gr.Chatbot(value=[], elem_id='chatbot', height=530)

        with gr.Row():
            with gr.Column(scale=0.60):
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your Question",
                container=False)

            with gr.Column(scale=0.20):
                submit_button = gr.Button('Send')

            with gr.Column(scale=0.20):
                uploaded_pdf = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])
                

        return demo, chat_history, text_input, submit_button, uploaded_pdf

if __name__ == '__main__':
    demo, chatbot, text_input, submit_button, uploaded_pdf = create_UI()
    demo.queue()
    demo.launch(share=True)