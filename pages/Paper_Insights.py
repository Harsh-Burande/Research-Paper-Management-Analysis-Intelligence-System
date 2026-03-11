import streamlit as st

st.title("Research Paper Insights")

metadata = st.session_state.get("paper_metadata", None)

if metadata:

    st.subheader("Title")
    st.write(metadata["title"])

    st.subheader("Authors")
    st.write(metadata["authors"])

    st.subheader("Abstract")
    st.write(metadata["abstract"])

    st.subheader("Published Year")
    st.write(metadata["year"])

    st.sidebar.title("Referenced Papers")

    for ref in metadata["references"]:
        st.sidebar.write(ref)

else:
    st.warning("Please upload a research paper on Page 1 first.")






