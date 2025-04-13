# main.py - Main application file for the Multilingual Voice Assistant

import os
import time
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import asyncio
import edge_tts
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import pipeline as hf_pipeline
import whisper
import fasttext
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class MultilingualVoiceAssistant:
    def __init__(self, knowledge_dir="knowledge_base", sample_rate=16000):
        """
        Initialize the voice assistant with all required components
        
        Args:
            knowledge_dir: Directory containing knowledge base documents
            sample_rate: Audio sample rate for recording and playback
        """
        self.sample_rate = sample_rate
        self.knowledge_dir = knowledge_dir
        
        # Ensure the knowledge directory exists
        os.makedirs(knowledge_dir, exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        logger.info("Initializing models, this may take a moment...")
        
        # Initialize Speech-to-Text model (Whisper)
        self.load_stt_model()
        
        # Initialize Language Detection model
        self.load_language_detector()
        
        # Initialize RAG components
        self.setup_rag_pipeline()
        
        logger.info("Voice assistant initialized and ready!")

    def load_stt_model(self):
        """Load the Whisper model for speech recognition"""
        logger.info("Loading Whisper model...")
        self.stt_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")

    def load_language_detector(self):
        """Load the language detection model"""
        logger.info("Loading language detection model...")
        # Download the language detection model if not already present
        if not os.path.exists("lid.176.bin"):
            os.system("wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        
        self.lang_detector = fasttext.load_model("lid.176.bin")
        logger.info("Language detection model loaded successfully")

    def setup_rag_pipeline(self):
        """Set up the RAG pipeline with embedding model, vector store, and LLM"""
        logger.info("Setting up RAG pipeline...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Set up vector store if documents exist
        if os.path.exists(self.knowledge_dir) and any(os.scandir(self.knowledge_dir)):
            # Load documents
            loader = DirectoryLoader(self.knowledge_dir, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            logger.info(f"Vector store created with {len(texts)} document chunks")
        else:
            # Create an empty vector store
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
            logger.info("Created empty vector store. Add documents to your knowledge base.")
        
        # Initialize the LLM
        self.setup_llm()
        
        # Create the retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        logger.info("RAG pipeline setup complete")

    def setup_llm(self):
        """Set up the language model for response generation"""
        logger.info("Loading language model...")
        
        # Load model and tokenizer
        model_id = "microsoft/phi-2"  # A smaller model that works on CPU
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Create LangChain pipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Language model loaded successfully")

    def record_audio(self, duration=5):
        """
        Record audio from the microphone
        
        Args:
            duration: Length of recording in seconds
            
        Returns:
            Path to the recorded audio file
        """
        logger.info(f"Recording audio for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Save recording to file
        timestamp = int(time.time())
        audio_path = f"temp/recording_{timestamp}.wav"
        recording = recording * 32767 / np.max(np.abs(recording))
        write(audio_path, self.sample_rate, recording.astype(np.int16))
        
        logger.info(f"Audio recorded and saved to {audio_path}")
        return audio_path

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Containing transcribed text and detected language
        """
        logger.info("Transcribing audio...")
        
        # Transcribe audio using Whisper
        result = self.stt_model.transcribe(audio_path)
        
        logger.info(f"Transcription complete. Detected language: {result['language']}")
        return {
            "text": result["text"],
            "language": result["language"]
        }

    def detect_language(self, text):
        """
        Detect the language of the input text
        
        Args:
            text: Input text
            
        Returns:
            str: Language code
        """
        prediction = self.lang_detector.predict(text.replace("\n", " "))
        lang_code = prediction[0][0].replace("__label__", "")
        return lang_code

    def process_query(self, query):
        """
        Process a query through the RAG pipeline
        
        Args:
            query: Text query
            
        Returns:
            str: Generated response
        """
        logger.info("Processing query through RAG pipeline...")
        
        # Handle the case of an empty vector store
        if self.vectorstore._collection.count() == 0:
            return "I don't have any knowledge base documents yet. Please add some documents to help me answer questions."
        
        # Process the query through the RAG pipeline
        result = self.qa_chain({"query": query})
        
        # Extract and return the response
        response = result["result"]
        logger.info("Query processed successfully")
        return response

    async def text_to_speech(self, text, language_code, output_path):
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            language_code: Language code for TTS
            output_path: Path to save the audio file
            
        Returns:
            str: Path to the generated audio file
        """
        logger.info(f"Converting text to speech in language: {language_code}")
        
        # Map language codes to Edge TTS voices
        voice_map = {
            "en": "en-US-ChristopherNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural",
            "it": "it-IT-DiegoNeural",
            "pt": "pt-BR-AntonioNeural",
            "ru": "ru-RU-DmitryNeural",
            "zh": "zh-CN-YunxiNeural",
            "ja": "ja-JP-KeitaNeural",
            "ko": "ko-KR-InJoonNeural",
            "ar": "ar-SA-HamedNeural",
            "hi": "hi-IN-MadhurNeural"
        }
        
        # Get the first two characters of the language code
        lang_prefix = language_code.split("-")[0] if "-" in language_code else language_code[:2]
        
        # Select voice or default to English
        voice = voice_map.get(lang_prefix, "en-US-ChristopherNeural")
        
        # Create TTS communicator
        communicate = edge_tts.Communicate(text, voice)
        
        # Generate speech
        await communicate.save(output_path)
        logger.info(f"Speech generated and saved to {output_path}")
        return output_path

    def play_audio(self, audio_path):
        """
        Play audio file
        
        Args:
            audio_path: Path to the audio file
        """
        logger.info(f"Playing audio from {audio_path}")
        
        # Load and play audio
        data, fs = sf.read(audio_path)
        sd.play(data, fs)
        sd.wait()
        logger.info("Audio playback complete")

    def add_document_to_knowledge_base(self, file_path, content):
        """
        Add a document to the knowledge base
        
        Args:
            file_path: Path where the document will be saved
            content: Content of the document
        """
        logger.info(f"Adding document to knowledge base: {file_path}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the document
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Load the document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split the document
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(texts)
        
        logger.info(f"Document added to knowledge base with {len(texts)} chunks")

    async def process_voice_query(self):
        """
        Process a complete voice query, from audio input to audio output
        
        Returns:
            dict: Result containing transcription, response, and language
        """
        # Record audio
        audio_path = self.record_audio(duration=5)
        
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        query_text = transcription["text"]
        language = transcription["language"]
        
        # Double-check language with fasttext
        detected_lang = self.detect_language(query_text)
        
        # Use the more confident language detection
        final_language = detected_lang if detected_lang else language
        
        logger.info(f"Transcribed query: '{query_text}' in language: {final_language}")
        
        # Process the query
        response = self.process_query(query_text)
        
        # Convert response to speech
        tts_output_path = f"temp/response_{int(time.time())}.mp3"
        await self.text_to_speech(response, final_language, tts_output_path)
        
        # Play the response
        self.play_audio(tts_output_path)
        
        return {
            "query": query_text,
            "response": response,
            "language": final_language
        }

async def main():
    """Main function to initialize and run the voice assistant"""
    print("Initializing Multilingual Voice Assistant with RAG...")
    assistant = MultilingualVoiceAssistant()
    
    print("\nVoice Assistant initialized! You can now interact with it.")
    print("Type 'exit' to quit, 'help' for commands, or press Enter to start a voice query.")
    
    while True:
        command = input("\nPress Enter to speak, or type a command: ").strip().lower()
        
        if command == "exit":
            print("Exiting voice assistant. Goodbye!")
            break
        elif command == "help":
            print("\nAvailable commands:")
            print("  help - Show this help message")
            print("  exit - Exit the application")
            print("  add - Add a document to the knowledge base")
            print("  [Enter] - Start a voice query")
        elif command == "add":
            # Add a document to the knowledge base
            doc_name = input("Enter document name (e.g., science/physics.txt): ")
            if not doc_name.endswith(".txt"):
                doc_name += ".txt"
            
            file_path = os.path.join(assistant.knowledge_dir, doc_name)
            print(f"Enter document content (finish with an empty line):")
            
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            content = "\n".join(lines)
            assistant.add_document_to_knowledge_base(file_path, content)
            print(f"Document added to knowledge base: {file_path}")
        else:
            # Process voice query
            try:
                result = await assistant.process_voice_query()
                print(f"\nQuery: {result['query']}")
                print(f"Response: {result['response']}")
                print(f"Language: {result['language']}")
            except Exception as e:
                print(f"Error processing voice query: {e}")

if __name__ == "__main__":
    asyncio.run(main())