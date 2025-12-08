import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

API_URL = "https://adib241-spotify-backend.hf.space"

st.set_page_config(page_title="♫ SpotifyXplore", page_icon="", layout="wide")

st.markdown("""
<style>
    .stMetric { background-color: #08062E; padding: 10px; border-radius: 5px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
</style>
""", unsafe_allow_html=True)

# Sidebar & Connection 
with st.sidebar:
    st.header(" Server Connection")
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            status = response.json()
            st.success(f"Connected to Neural Engine")
            st.metric("Status", status["status"].upper())
            st.metric("Tracks Indexed", f"{status['tracks_loaded']:,}")
            st.caption(f"Backend Device: {status.get('device', 'unknown')}")
        else:
            st.error("Server Error")
    except:
        st.error("❌ Backend Offline")
        st.warning("Please run: uvicorn 2_backend_api:app --reload")
        st.stop()

# Data Fetching 
@st.cache_data
def get_genres():
    try:
        return requests.get(f"{API_URL}/genres").json()["genres"]
    except:
        return []

genres_list = get_genres()

# Main UI
st.title("♫ Music Recommender")

tab1, tab2 = st.tabs(["🔍 Similarity Engine", "🎛️ Playlist Curator"])

with tab1:
    st.subheader("Neural Vector Search")
    
    # Search
    search_query = st.text_input("Search for a seed song", placeholder="e.g. Gravity")
    selected_song = None
    if search_query:
        try:
            with st.spinner("Searching..."):
                resp = requests.get(f"{API_URL}/search", params={"q": search_query, "limit": 20})
                if resp.status_code == 200:
                    matches = resp.json()["matches"]
                    if matches:
                        selected_song = st.selectbox("Select exact match:", matches)
                    else:
                        st.warning("No matches found.")
        except: pass

    st.divider()
    num_recs = st.slider("Recommendations Count", 5, 50, 10, key="slider_sim")
    
    if st.button("Find Similar Songs", type="primary", disabled=not selected_song):
        start_time = time.time()
        try:
            resp = requests.post(f"{API_URL}/recommend/similar", json={"song_name": selected_song, "k": num_recs})
            latency = (time.time() - start_time) * 1000
            
            if resp.status_code == 200:
                recs = resp.json()["recommendations"]
                st.success(f"Retrieved {len(recs)} matches in {latency:.0f}ms")
                
                df = pd.DataFrame(recs)
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.dataframe(df[['name', 'artist', 'genre', 'score']], use_container_width=True)
                with c2:
                    fig = go.Figure(go.Bar(x=df['score'], y=df['name'], orientation='h', marker=dict(color=df['score'], colorscale='Viridis')))
                    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# PLAYLIST CURATOR
with tab2:
    st.subheader("Feature-Based Curator")
    
    col_filters, col_sliders = st.columns([1, 1])
    
    with col_filters:
        sel_genres = st.multiselect("Filter by Genre", genres_list)
        limit = st.slider("Playlist Size", 20, 100, 50)
        
    with col_sliders:
        energy = st.slider("Energy Range", 0.0, 1.0, (0.4, 0.9))
        dance = st.slider("Danceability Range", 0.0, 1.0, (0.5, 1.0))
        tempo = st.slider("Tempo (BPM)", 50, 220, (90, 140))

    if st.button("✨ Curate Playlist", type="primary"):
        payload = {
            "genres": sel_genres,
            "energy_min": energy[0], "energy_max": energy[1],
            "dance_min": dance[0], "dance_max": dance[1],
            "tempo_min": tempo[0], "tempo_max": tempo[1],
            "limit": limit
        }
        
        start_time = time.time()
        try:
            with st.spinner("Filtering 1M+ tracks..."):
                resp = requests.post(f"{API_URL}/recommend/curate", json=payload)
                
            latency = (time.time() - start_time) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                playlist = data["playlist"]
                
                if playlist:
                    st.success(f"Generated {len(playlist)} tracks in {latency:.0f}ms")
                    
                    df_pl = pd.DataFrame(playlist)
                    
                    # Layout: Table left, Radar Chart right
                    pc1, pc2 = st.columns([2, 1])
                    
                    with pc1:
                        st.dataframe(
                            df_pl[['name', 'artist', 'genre', 'year', 'energy', 'danceability', 'tempo']], 
                            use_container_width=True, height=500
                        )
                        
                    with pc2:
                        st.markdown("### Playlist Vibe")
                        avg_stats = df_pl[['energy', 'danceability']].mean()
                        avg_stats['tempo (scaled)'] = df_pl['tempo'].mean() / 200.0
                        
                        r_data = pd.DataFrame({
                            'r': avg_stats.values,
                            'theta': ['Energy', 'Danceability', 'Tempo']
                        })
                        
                        fig = px.line_polar(r_data, r='r', theta='theta', line_close=True, range_r=[0, 1])
                        fig.update_layout(title="Average Audio Signature")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No songs matched your specific criteria. Try widening the sliders!")
            else:
                st.error("API Error")
                
        except Exception as e:

            st.error(f"Connection Failed: {e}")

