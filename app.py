import streamlit as st
from rag import process_documents, ask_question


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------

st.title("👩‍💼 HR Policy Assistant")

st.markdown(
    """
    **Ask questions about your company's HR policies.**

    Upload your HR policy PDFs and the AI will answer questions
    using information from those documents.
    """
)

st.divider()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📄 HR Documents")

    uploaded_files = st.file_uploader(
        "Upload HR policy PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"{len(uploaded_files)} document(s) uploaded.")

        if st.button("🔄 Process Documents", use_container_width=True):

            with st.spinner("Reading and indexing policies..."):

                try:
                    chunks = process_documents(uploaded_files)

                    st.session_state["documents_processed"] = True

                    st.success(
                        f"Successfully indexed {chunks} policy chunks."
                    )

                except Exception as e:

                    st.error(f"Error: {str(e)}")


    st.divider()

    st.markdown("### 💡 Example Questions")

    st.markdown(
        """
        - How many annual leaves are allowed?
        - What is the work from home policy?
        - How long is maternity leave?
        - What is the late arrival policy?
        - Can employees carry forward annual leave?
        """
    )


# -----------------------------
# Main Chat
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask something about the HR policies..."
)


if question:

    if not st.session_state.get("documents_processed", False):

        st.warning(
            "Please upload and process your HR policy documents first."
        )

    else:

        # Display user question
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching HR policies..."):

                try:

                    answer, sources = ask_question(question)

                    st.markdown(answer)

                    # Show sources
                    if sources:

                        with st.expander("📚 Sources"):

                            for source in sources:

                                st.markdown(
                                    f"- {source}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {str(e)}"
                    )
