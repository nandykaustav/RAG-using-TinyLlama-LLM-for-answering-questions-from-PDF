from UI import create_UI
from model import RAGChatBot

# Create Gradio interface
demo, chat_history, txt, submit_button, uploaded_pdf = create_UI()

# Create PDFChatBot instance
rag_chatbot = RAGChatBot()

# Set up event handlers
with demo:
    uploaded_pdf.click(rag_chatbot.upload_click)
    uploaded_pdf.upload(rag_chatbot.file_upload)
    # Event handler for submitting text and generating response
    submit_button.click(rag_chatbot.append_to_history, inputs=[chat_history, txt], outputs=[chat_history], queue=False).\
        success(rag_chatbot.generate_response, inputs=[chat_history, txt, uploaded_pdf], outputs=[chat_history, txt])


if __name__ == "__main__":
    demo.launch(share=True)