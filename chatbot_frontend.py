import streamlit as st
from chatbot_backend import (
    chatbot,
    retrive_all_threads,
    ingest_pdf,
    thread_has_document,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

st.set_page_config(page_title="LangGraph Chatbot with RAG", page_icon="🤖", layout="wide")

### Utility Functions
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    return state.values.get('messages', [])


### Session Setup
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()

add_thread(st.session_state['thread_id'])
current_thread = str(st.session_state['thread_id'])


### Sidebar UI
st.sidebar.title("🤖 LangGraph Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")

# --- RAG / Document Upload Section ---
st.sidebar.subheader("📄 Document Context (RAG)")

doc_meta = thread_document_metadata(current_thread)
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key=f"uploader_{current_thread}",
)

if uploaded_file is not None:
    if not thread_has_document(current_thread) or doc_meta.get("filename") != uploaded_file.name:
        with st.sidebar.status("🔄 Ingesting & indexing PDF...", expanded=True) as status:
            try:
                summary = ingest_pdf(
                    uploaded_file.getvalue(),
                    thread_id=current_thread,
                    filename=uploaded_file.name,
                )
                status.update(label="✅ Indexed successfully!", state="complete", expanded=False)
                st.sidebar.success(
                    f"**{summary['filename']}** indexed!\n\n"
                    f"• {summary['documents']} pages\n• {summary['chunks']} chunks"
                )
                st.rerun()
            except Exception as e:
                status.update(label="❌ Ingestion failed", state="error", expanded=True)
                st.sidebar.error(f"Error indexing PDF: {e}")

if thread_has_document(current_thread):
    info = thread_document_metadata(current_thread)
    st.sidebar.info(
        f"📎 **Attached Document:**\n\n"
        f"**File:** `{info.get('filename')}`\n\n"
        f"• **Pages:** {info.get('documents', 0)}\n"
        f"• **Chunks:** {info.get('chunks', 0)}"
    )
else:
    st.sidebar.caption("No document attached to this thread yet. Upload a PDF above to ask questions from it.")

st.sidebar.markdown("---")

# --- Chat Threads Section ---
st.sidebar.subheader("💬 My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    is_active = (thread_id == st.session_state['thread_id'])
    label = f"{'👉 ' if is_active else ''}Chat: {str(thread_id)[:8]}..."
    
    if st.sidebar.button(label, key=f"btn_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({'role': 'user', 'content': msg.text})
            elif isinstance(msg, AIMessage) and msg.text and msg.text.strip():
                temp_messages.append({'role': 'assistant', 'content': msg.text})

        st.session_state['message_history'] = temp_messages
        st.rerun()


# --- Main Chat Area ---
st.title("LangGraph Multi-Tool Chatbot with RAG")

# Show current thread info banner
if thread_has_document(current_thread):
    st.caption(f"Thread: `{current_thread[:8]}...` | 📄 Context: `{doc_meta.get('filename')}`")
else:
    st.caption(f"Thread: `{current_thread[:8]}...` | 🌐 Tools: Web Search, Stock Price, Calculator, PDF RAG")

# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input("Type your message or ask questions about your PDF...")

if user_input:
    # 1. Add user message to history & display
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 2. Configuration for LangGraph
    CONFIG = {
        "configurable": {'thread_id': st.session_state['thread_id']},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # 3. Stream assistant response
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Update status container when tools run
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if tool_name == "rag_tool":
                        display_name = "📄 Searching Document (RAG)"
                    elif tool_name == "search_tool":
                        display_name = "🌐 Searching Web"
                    elif tool_name == "calculator":
                        display_name = "🧮 Calculating"
                    elif tool_name == "get_stock_price":
                        display_name = "📈 Fetching Stock Price"
                    else:
                        display_name = f"🔧 Using `{tool_name}`"

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"{display_name} …", expanded=True)
                    else:
                        status_holder["box"].update(
                            label=f"{display_name} …",
                            state="running",
                            expanded=True,
                        )

                # Stream assistant content
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.text:
                        yield message_chunk.text

        ai_message = st.write_stream(ai_only_stream())

        # Complete tool status box if a tool was used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
