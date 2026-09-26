import os
import streamlit as st

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate


# ==============================
# Load API Key
# ==============================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF API key not found!")
    st.stop()


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(
    page_title="Himanshu AI",
    page_icon="🤖"
)

st.title("🤖 Himanshu's Personal AI Assistant")
st.caption("Built with LangChain + Hugging Face")


# ==============================
# Hugging Face Client
# ==============================

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# ==============================
# LangChain Prompt
# ==============================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are Himanshu's personal AI assistant.

        Be friendly, helpful and concise.

        Explain technical concepts in simple language.

        Himanshu is an Electronics and Communication Engineering
        student interested in AI, GenAI, LangChain, Agentic AI,
        web development and programming.

        For coding questions, provide clear explanations and
        working code.

        If you don't know something, say that you don't know.
        """
    ),
    (
        "human",
        "{question}"
    )
])
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================
# Chat History
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display history

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================
# User Input
# ==============================

question = st.chat_input("Ask me anything...")


if question:

    # Show user message

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # Create prompt

    formatted_prompt = prompt.invoke({
        "question": question
    })


    # Convert LangChain messages
    # into Hugging Face messages

    hf_messages = []

    for message in formatted_prompt.messages:

        if message.type == "system":
            role = "system"

        elif message.type == "human":
            role = "user"

        else:
            role = "assistant"

        hf_messages.append({
            "role": role,
            "content": message.content
        })


    # ==============================
    # Generate Response
    # ==============================

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                          model="meta-llama/Llama-3.1-8B-Instruct",
                            messages=hf_messages,
                            max_tokens=512,
                            temperature=0.7
                                        )

            answer = response.choices[0].message.content

        st.markdown(answer)


    # Save response

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })