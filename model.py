import fitz
import gradio as gr
from PIL import Image
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class RAGChatBot:
    def __init__(self):

        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.processed = False
        self.page = 0
        self.chat_history = []
        self.prompt = None
        self.rawFile=None
        self.document = None
        self.embeddings = None
        self.vectordb = None
        self.model = None
        self.pipeline = None
        self.chain = None

    def load_file(self, file):

        #Load the PDF file uploaded

        loader = PyPDFLoader(file.name)
        self.rawFile=loader.load()
    def split_text(self):

        #Split the uploaded document into chunks so that it is easier for the LLM to look up the relevant part of the document

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.document = text_splitter.split_documents(self.rawFile)

    def append_to_history(self, history, text):

        #Appending the entered question into the chat history

        if not text:
            raise gr.Error('Enter text')
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history

    def initialize_embeddings(self):
        modelPath = "BAAI/bge-small-en-v1.5"

        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        model_kwargs = {'device':'cuda'}

        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': False}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )
        self.embeddings = embeddings

    def initialize_vectordb(self):

        #Initializing FAISS - the library for doing vector similarity search
 
        self.vectordb = FAISS.from_documents(self.document, self.embeddings)


    def initialize_model(self):

        # Load the tokenizer associated with the specified model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding=True, truncation=True, max_length=512)

        # Define a question-answering pipeline using the model and tokenizer
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            max_new_tokens=100,  # max number of tokens to generate in the output
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1
        )

    def create_pipeline(self):

        # Wrapping the llm in a HuggingFacePipeline

        self.pipeline = HuggingFacePipeline(
            pipeline=self.model
        )

    def create_prompt_template(self):

        # Create a prompt template to prompt the LLM. This can be tuned to optimize the answers of the model
        
        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:
        """

        self.prompt = PromptTemplate(
         template=prompt_template, input_variables=["context", "question"]
        )

    def create_retreiver_chain(self):
        # Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 2})

        # Create a question-answering instance (qa) using the RetrievalQA class.
        # It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
        self.chain = RetrievalQA.from_chain_type(
            llm=self.pipeline,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def run_processes(self, file):

        # Run all steps of the process

        self.load_file(file)
        self.split_text()
        self.initialize_embeddings()
        self.initialize_vectordb()
        self.initialize_model()
        self.create_pipeline()
        self.create_prompt_template()
        self.create_retreiver_chain()

    def generate_response(self, history, query, file):
        
        if not query:
            raise gr.Error(message='Enter your question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.run_processes(file)  #Run all steps of the process
            self.processed = True

        # Run the chain and generate the result

        result = self.chain({"query": query, 'chat_history': self.chat_history}, return_only_outputs=True)
        self.chat_history.append((query, result["result"]))

        for char in result['result']:
            history[-1][-1] += char
        return history, " "

    def file_upload(self):

        # Alert for successful upload
    
        gr.Info("File uploaded successfully")

    def upload_click(self):

        # Alert for upload button click

        gr.Info("Uploading file. Please wait till you get success message")