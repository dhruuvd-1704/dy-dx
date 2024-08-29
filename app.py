import streamlit as st
import tempfile


def landing_page():
    #css
    st.markdown("""
        <style>
        .main {
            background-color: black;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .title {
            font-size: 3.5em;
            color: #FF4B4B;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 2em;
            color: white;
            margin-bottom: 40px;
        }
        .get-started-button {
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 1.5em;
            border-radius: 5px;
            cursor: pointer;
        }
        .get-started-button:hover {
            background-color: #ff3333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main">
            <div class="title">Deep Fake Video Detection</div>
            <div class="subtitle">Protecting authenticity in the digital age</div>
          
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Get Started"):
        st.session_state.page = "functionality_page"


def functionality_page():
    st.title("Deep Fake Video Detection")
    
    if st.button("Back"):
        st.session_state.page = "landing_page"

   
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
   
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        
        st.video(video_path)
        
        if st.button("Detect"):
            st.success("This is a placeholder. Replace with actual detection logic.")


def main():
    if "page" not in st.session_state:
        st.session_state.page = "landing_page"
    
    if st.session_state.page == "landing_page":
        landing_page()
    elif st.session_state.page == "functionality_page":
        functionality_page()

if __name__ == "__main__":
    main()
