import yt_dlp
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# 🎨 Streamlit Page Config
st.set_page_config(page_title="📜 AI Summarizer: YouTube & Web", page_icon="🔥", layout="wide")

# 🏆 Header Design
st.markdown("""
    <style>
    .stTextInput>div>div>input {border-radius: 10px;}
    .stButton>button {background-color: #ff4b4b; color: white; font-size: 18px;}
    .stSpinner>div>div {color: #ff4b4b;}
    </style>
""", unsafe_allow_html=True)

st.title("📜 AI Summarizer: YouTube & Web")
st.subheader("🚀 Get AI-powered summaries from **YouTube videos** & **Websites** instantly!")

# 🔑 Sidebar: API Key
with st.sidebar:
    st.subheader("🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.markdown("📌 **Note:** Works best with YouTube or **text-heavy** webpages.")

# 🎯 Input: URL (YouTube or Website)
generic_url = st.text_input("🎯 Enter YouTube or Website URL")

# 📽️ Function to Get YouTube Details using yt-dlp
def get_youtube_details(url):
    try:
        ydl_opts = {"quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown Title"),
                "channel": info.get("uploader", "Unknown Channel"),
                "date": info.get("upload_date", "N/A")
            }
    except Exception as e:
        return {"title": "Error fetching title", "channel": "N/A", "date": "N/A"}

# 🚀 Load LLM Model (gemma2-9b-it)
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# 📌 Summarization Prompt Template
prompt_template = """
Summarize the following content in 250-300 words with key highlights:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# 🎯 Button to Summarize Content
if st.button("🔍 Generate Summary"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("⚠️ Please provide both the Groq API Key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("🚨 Invalid URL! Enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("🚀 Processing... Please wait..."):
                # Check if it's a YouTube URL
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    yt_details = get_youtube_details(generic_url)
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0"})
                
                # Load content
                docs = loader.load()

                # ⏳ Progress Bar
                progress = st.progress(20)

                # Run Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                progress.progress(100)

                # 🎉 Display Summary
                st.success("✅ Summary Generated Successfully!")

                if "youtube.com" in generic_url:
                    st.write(f"📽️ **Video Title:** {yt_details['title']}")
                    st.write(f"📺 **Channel:** {yt_details['channel']}")
                    st.write(f"🗓️ **Upload Date:** {yt_details['date']}")

                st.write("### ✍️ AI Summary:")
                st.info(output_summary)

                # 🔻 Download Summary Button
                st.download_button(label="📥 Download Summary", data=output_summary, file_name="summary.txt", mime="text/plain")

        except Exception as e:
            st.error(f"❌ Error: {e}")
