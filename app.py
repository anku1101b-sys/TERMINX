import streamlit as st
import numpy as np
import faiss
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# TERMINX — AI Knowledge Intelligence
# ---------------------------------------------------------

st.set_page_config(
    page_title="TERMINX | AI Knowledge Intelligence",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #080b12;
    color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #0b1220);
    border: 1px solid #263247;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 5px;
}

.hero p {
    color: #9ca9bd;
    font-size: 18px;
}

.card {
    background: #101621;
    border: 1px solid #263247;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
}

.source {
    background: #0c111b;
    border-left: 3px solid #6ea8fe;
    padding: 14px;
    border-radius: 8px;
    margin-top: 10px;
}

.answer {
    background: #111a2b;
    border: 1px solid #30435f;
    padding: 22px;
    border-radius: 16px;
    line-height: 1.7;
    font-size: 17px;
}

.small {
    color: #8d9bb0;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# KNOWLEDGE BASE
# ---------------------------------------------------------

knowledge_base = [
    {
        "title": "Artificial Intelligence",
        "text": """
Artificial Intelligence is the field of computer science focused on creating
systems that can perform tasks that normally require human intelligence.
These tasks include reasoning, learning, perception, decision making and
language understanding.
"""
    },
    {
        "title": "Machine Learning",
        "text": """
Machine Learning is a branch of artificial intelligence in which computers
learn patterns from data instead of being explicitly programmed for every
task. Common types include supervised learning, unsupervised learning and
reinforcement learning.
"""
    },
    {
        "title": "Deep Learning",
        "text": """
Deep Learning is a subset of machine learning that uses artificial neural
networks with multiple layers. Deep learning is widely used for image
recognition, speech recognition, natural language processing and computer
vision.
"""
    },
    {
        "title": "Natural Language Processing",
        "text": """
Natural Language Processing, or NLP, allows computers to process and
understand human language. Applications include chatbots, translation,
sentiment analysis, text classification and question answering.
"""
    },
    {
        "title": "Retrieval-Augmented Generation",
        "text": """
Retrieval-Augmented Generation, commonly called RAG, combines information
retrieval with language generation. A RAG system first retrieves relevant
information from a knowledge base and then uses that information to produce
a grounded answer.
"""
    },
    {
        "title": "Vector Databases",
        "text": """
Vector databases store numerical representations called embeddings.
Embeddings represent the semantic meaning of text. Similarity search can
then be used to find information that is semantically related to a query.
FAISS is a library that provides efficient similarity search for vectors.
"""
    },
    {
        "title": "Computer Science",
        "text": """
Computer Science is the study of computation, algorithms, software,
programming languages, data structures, artificial intelligence and computer
systems. It provides the foundations for modern software and technology.
"""
    }
]

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = knowledge_base.copy()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def create_chunks(documents):
    chunks = []

    for doc in documents:
        text = re.sub(r"\s+", " ", doc["text"]).strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)

        current = []

        for sentence in sentences:
            if sentence.strip():
                current.append(sentence.strip())

            if len(current) >= 3:
                chunks.append({
                    "text": " ".join(current),
                    "source": doc["title"]
                })
                current = []

        if current:
            chunks.append({
                "text": " ".join(current),
                "source": doc["title"]
            })

    return chunks

# ---------------------------------------------------------
# BUILD VECTOR INDEX
# ---------------------------------------------------------

def build_index():
    chunks = create_chunks(st.session_state.documents)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.session_state.chunks = chunks
    st.session_state.index = index

if st.session_state.index is None:
    build_index()

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------

def search_knowledge(query, top_k=4):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = st.session_state.index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": st.session_state.chunks[idx]["text"],
                "source": st.session_state.chunks[idx]["source"],
                "score": float(score)
            })

    return results

# ---------------------------------------------------------
# ANSWER GENERATION
# ---------------------------------------------------------

def generate_answer(query, results):

    if not results:
        return "I could not find relevant information in the knowledge base."

    # Use the strongest retrieved sources first
    best_results = results[:2]

    answer_parts = []

    for result in best_results:
        text = result["text"].strip()

        if text and text not in answer_parts:
            answer_parts.append(text)

    if not answer_parts:
        return "I could not find enough relevant information."

    return " ".join(answer_parts)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>⚡ TERMINX</h1>
    <p>AI Knowledge Intelligence • Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## ⚡ TERMINX")

    st.markdown(
        "Semantic knowledge retrieval powered by embeddings and FAISS."
    )

    st.divider()

    st.markdown("### Knowledge Base")

    st.metric(
        "Documents",
        len(st.session_state.documents)
    )

    st.metric(
        "Knowledge Chunks",
        len(st.session_state.chunks)
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "📥 Index PDF",
            use_container_width=True
        ):

            reader = PdfReader(uploaded_file)

            extracted_text = ""

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"

            if extracted_text.strip():

                st.session_state.documents.append({
                    "title": uploaded_file.name,
                    "text": extracted_text
                })

                build_index()

                st.success(
                    f"Indexed {uploaded_file.name}"
                )

            else:
                st.error(
                    "No readable text was found in the PDF."
                )

    st.divider()

    if st.button(
        "🔄 Reset Knowledge Base",
        use_container_width=True
    ):

        st.session_state.documents = knowledge_base.copy()
        st.session_state.messages = []

        build_index()

        st.rerun()

    if st.button(
        "🧹 New Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Documents",
        len(st.session_state.documents)
    )

with col2:
    st.metric(
        "Indexed Chunks",
        len(st.session_state.chunks)
    )

with col3:
    st.metric(
        "Vector Search",
        "FAISS"
    )

st.markdown("## Ask TERMINX")

question = st.chat_input(
    "Ask something about AI, ML, RAG, NLP..."
)

# ---------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------

if not st.session_state.messages:

    st.markdown("### 💡 Try a question")

    suggestions = [
        "How does RAG work?",
        "What is machine learning?",
        "What are vector databases?",
        "What is deep learning?",
        "What is NLP?"
    ]

    cols = st.columns(len(suggestions))

    for col, suggestion in zip(cols, suggestions):

        with col:

            if st.button(
                suggestion,
                use_container_width=True
            ):
                question = suggestion

# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    results = search_knowledge(
        question,
        top_k=4
    )

    answer = generate_answer(
        question,
        results
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "results": results
        }
    )

# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):

            st.markdown(
                f'<div class="answer">{message["content"]}</div>',
                unsafe_allow_html=True
            )

            results = message.get(
                "results",
                []
            )

            with st.expander(
                "🔎 RAG Retrieval Trace"
            ):

                st.markdown(
                    "**1. Query → Embedding**  \n"
                    "The question is converted into a numerical vector."
                )

                st.markdown(
                    "**2. Retrieve → FAISS**  \n"
                    "FAISS searches the knowledge base for semantically similar chunks."
                )

                st.markdown(
                    "**3. Ground → Answer**  \n"
                    "TERMINX builds the answer using the retrieved information."
                )

                st.divider()

                st.markdown("### Retrieved Sources")

                for i, result in enumerate(results, 1):

                    st.markdown(
                        f"""
                        <div class="source">
                            <b>{i}. {result["source"]}</b><br>
                            <span class="small">
                                Relevance: {result["score"]:.2f}
                            </span>
                            <br><br>
                            {result["text"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#6f7c91;">
        TERMINX • AI Knowledge Intelligence
        <br>
        <span class="small">
        Semantic Retrieval • Embeddings • FAISS • RAG
        </span>
    </div>
    """,
    unsafe_allow_html=True
)