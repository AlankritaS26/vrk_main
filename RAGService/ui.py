"""
RAGService Web UI
=================
Admin panel for browsing, indexing, and searching the RAG vector store.
Talks to the RAGService API — point it at any host:port.

Start:  streamlit run ui.py --server.port 8601
"""
import io
import requests
import streamlit as st

st.set_page_config(
    page_title="RAGService Admin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; }
    header[data-testid="stHeader"] { height:0; min-height:0; }
    div[data-testid="stDecoration"]{ display:none; }
    .score-high  { color:#00843D; font-weight:700; }
    .score-med   { color:#E65100; font-weight:700; }
    .score-low   { color:#9E9E9E; font-weight:700; }
    .chunk-card  {
        background:#F8F9FA; border-left:4px solid #0070C0;
        border-radius:6px; padding:10px 14px; margin-bottom:8px;
        font-size:0.9em;
    }
    .chunk-card.high { border-left-color:#00843D; }
    .chunk-card.med  { border-left-color:#E65100; }
    .tag {
        display:inline-block; background:#E3F2FD; color:#0070C0;
        border-radius:10px; padding:1px 8px; margin:2px; font-size:0.8em;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────

def _api(base_url: str) -> "API":
    return API(base_url)


class API:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def _get(self, path: str, **kw):
        return requests.get(f"{self.base}{path}", timeout=30, **kw)

    def _post(self, path: str, **kw):
        return requests.post(f"{self.base}{path}", timeout=120, **kw)

    def _delete(self, path: str):
        return requests.delete(f"{self.base}{path}", timeout=30)

    def health(self) -> dict:
        r = self._get("/health")
        r.raise_for_status()
        return r.json()

    def collections(self) -> list[str]:
        r = self._get("/v1/collections")
        r.raise_for_status()
        return r.json().get("collections", [])

    def stats(self, col: str) -> dict:
        r = self._get(f"/v1/collections/{col}")
        r.raise_for_status()
        return r.json()

    def files(self, col: str) -> list[str]:
        r = self._get(f"/v1/collections/{col}/files")
        r.raise_for_status()
        return r.json().get("files", [])

    def index_files(self, col: str, files: list, source: str) -> list[dict]:
        file_tuples = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
        r = self._post(f"/v1/collections/{col}/index/files",
                       files=file_tuples, data={"source": source})
        r.raise_for_status()
        return r.json()

    def index_text(self, col: str, text: str, source: str, metadata: dict) -> dict:
        r = self._post(f"/v1/collections/{col}/index/text",
                       json={"text": text, "source": source, "metadata": metadata})
        r.raise_for_status()
        return r.json()

    def search(self, col: str, query: str, k: int, filters: dict) -> list[dict]:
        r = self._post(f"/v1/collections/{col}/search",
                       json={"query": query, "k": k, "filters": filters})
        r.raise_for_status()
        return r.json()

    def delete_file(self, col: str, filename: str):
        self._delete(f"/v1/collections/{col}/files/{filename}").raise_for_status()

    def delete_collection(self, col: str):
        self._delete(f"/v1/collections/{col}").raise_for_status()


def _score_class(s: float) -> str:
    return "high" if s >= 0.75 else ("med" if s >= 0.50 else "")


def _score_color(s: float) -> str:
    return "score-high" if s >= 0.75 else ("score-med" if s >= 0.50 else "score-low")


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 RAGService Admin")
    st.caption("Index · Search · Manage")
    st.divider()

    # ── API connection ─────────────────────────────────────────────────
    st.subheader("🔌 API Connection")
    base_url = st.text_input("RAGService URL", value="http://localhost:8600",
                             help="Change this to point at a remote RAGService")

    api = _api(base_url)
    try:
        h = api.health()
        st.success(f"✅ Connected  |  `{h['embedding_provider']}` embeddings")
    except Exception as e:
        st.error(f"❌ Cannot reach API: {e}")
        st.info("Start the service: `./manage.sh start`")
        st.stop()

    st.divider()

    # ── Collection selector ────────────────────────────────────────────
    st.subheader("📂 Collection")
    try:
        existing_cols = api.collections()
    except Exception:
        existing_cols = []

    new_col_name = st.text_input("New collection name", placeholder="e.g. finance-q3")
    if st.button("➕ Create / Select", use_container_width=True) and new_col_name.strip():
        st.session_state["active_col"] = new_col_name.strip().lower().replace(" ", "-")

    if existing_cols:
        selected = st.selectbox("Or pick existing", ["—"] + existing_cols)
        if selected != "—":
            st.session_state["active_col"] = selected

    active_col = st.session_state.get("active_col", "")
    if active_col:
        st.success(f"Active: **{active_col}**")
    else:
        st.warning("No collection selected.")

    st.divider()
    if active_col:
        try:
            s = api.stats(active_col)
            st.metric("Chunks", s["total_chunks"])
            st.metric("Files",  len(s["files"]))
            st.caption(f"`{s['provider']}` · {s['dimensions']}D")
        except Exception:
            st.caption("(collection empty or not yet created)")


# ════════════════════════════════════════════════════════════════════════
# MAIN — requires an active collection
# ════════════════════════════════════════════════════════════════════════
if not active_col:
    st.title("🧠 RAGService Admin")
    st.info("👈 Enter a collection name in the sidebar to get started.")
    st.markdown("""
**Collections** are independent namespaces — one per project, app, or use-case.

| Tab | What you can do |
|-----|----------------|
| 📤 Index | Upload files or paste text to train the RAG |
| 🔍 Search | Run semantic search and see scored results |
| 📊 Browse | View all indexed files and browse raw chunks |
| 🗑️ Manage | Delete files or the entire collection |
    """)
    st.stop()

st.title(f"🧠 {active_col}")

tab_index, tab_search, tab_browse, tab_manage = st.tabs(
    ["📤 Index Data", "🔍 Search", "📊 Browse", "🗑️ Manage"]
)

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — INDEX
# ════════════════════════════════════════════════════════════════════════
with tab_index:
    st.subheader("Train the RAG — Add Data")

    sub_file, sub_text = st.tabs(["📎 Upload Files", "✏️ Paste Text"])

    # ── File upload ────────────────────────────────────────────────────
    with sub_file:
        st.markdown("Upload any supported files. They will be parsed, chunked, "
                    "embedded, and stored in **`" + active_col + "`**.")

        uploaded = st.file_uploader(
            "Drop files here",
            type=["pptx","ppt","docx","doc","pdf","xlsx","xls","csv","txt","md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        source_label = st.text_input(
            "Source / group label (optional)",
            placeholder="e.g. Finance, HR, Q3-2026",
            help="Attached as metadata to all chunks. Use it to filter searches later.",
        )

        if uploaded:
            st.markdown(f"**{len(uploaded)} file(s) ready:**  " +
                        "  ".join(f"`{f.name}`" for f in uploaded))

        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            do_index = st.button("⚡ Index Files", type="primary",
                                  disabled=not uploaded, use_container_width=True)
        with col_info:
            if not uploaded:
                st.info("Upload at least one file to enable indexing.")

        if do_index and uploaded:
            prog = st.progress(0.0)
            status = st.empty()
            results = []
            for i, f in enumerate(uploaded):
                status.markdown(f"**Indexing `{f.name}`…**")
                prog.progress((i) / len(uploaded))
                try:
                    res = api.index_files(active_col, [f], source_label)
                    results.extend(res)
                except Exception as e:
                    results.append({"message": f"ERROR: {e}", "added": 0})
            prog.progress(1.0)
            status.empty()

            total_added = sum(r.get("added", 0) for r in results)
            st.success(f"✅ Done! **{total_added}** new chunks indexed.")
            for r in results:
                icon = "✅" if r.get("added", 0) >= 0 and "ERROR" not in r["message"] else "❌"
                st.caption(f"{icon} {r['message']}")

    # ── Text input ─────────────────────────────────────────────────────
    with sub_text:
        st.markdown("Paste any text directly. Useful for indexing notes, meeting minutes, "
                    "database content, or API responses.")

        text_input = st.text_area(
            "Text to index",
            height=220,
            placeholder="Paste your text here. It will be chunked and indexed into the RAG.",
        )
        t_col1, t_col2 = st.columns(2)
        text_source = t_col1.text_input("Source label", placeholder="e.g. meeting-notes-2026-07")
        text_meta_raw = t_col2.text_input(
            "Extra metadata (JSON key:value pairs, optional)",
            placeholder='dept=Finance, quarter=Q3',
        )

        def _parse_meta(raw: str) -> dict:
            meta = {}
            for pair in raw.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    meta[k.strip()] = v.strip()
            return meta

        if st.button("⚡ Index Text", type="primary", disabled=not text_input.strip()):
            meta = _parse_meta(text_meta_raw)
            try:
                r = api.index_text(active_col, text_input, text_source, meta)
                st.success(f"✅ **{r['added']}** new chunks indexed.  {r['message']}")
            except Exception as e:
                st.error(f"❌ Failed: {e}")


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("Semantic Search")
    st.caption("Find the most relevant chunks using natural-language queries.")

    q_col, k_col = st.columns([5, 1])
    query   = q_col.text_input("Search query", placeholder="e.g. budget risks Q3 headcount")
    top_k   = k_col.number_input("Top K", min_value=1, max_value=50, value=10)

    # Optional filters
    with st.expander("🔧 Filters (optional)"):
        st.caption("Restrict results to specific metadata values.")
        fc1, fc2 = st.columns(2)
        f_source   = fc1.text_input("Source label filter", placeholder="Finance")
        f_filename = fc2.text_input("Filename filter", placeholder="Q3_report.pptx")

    filters = {}
    if f_source.strip():   filters["source"]   = f_source.strip()
    if f_filename.strip(): filters["filename"] = f_filename.strip()

    search_btn = st.button("🔍 Search", type="primary", disabled=not query.strip())

    if search_btn and query.strip():
        try:
            results = api.search(active_col, query, int(top_k), filters)
        except Exception as e:
            st.error(f"Search failed: {e}")
            results = []

        if not results:
            st.info("No results found. Try a different query or check that data has been indexed.")
        else:
            st.markdown(f"**{len(results)} results** for `{query}`" +
                        (f"  ·  filters: {filters}" if filters else ""))
            st.divider()

            for i, r in enumerate(results):
                score = r["score"]
                meta  = r["metadata"]
                cls   = _score_class(score)

                col_score, col_content = st.columns([1, 8])
                with col_score:
                    st.markdown(
                        f'<div style="text-align:center; padding-top:8px;">'
                        f'<span class="{_score_color(score)}">{score:.3f}</span>'
                        f'<br><small style="color:#888">score</small></div>',
                        unsafe_allow_html=True,
                    )
                with col_content:
                    # Metadata tags
                    tags = ""
                    for k, v in meta.items():
                        if k not in ("chunk_idx", "date_indexed") and v:
                            tags += f'<span class="tag">{k}: {v}</span>'

                    st.markdown(
                        f'<div class="chunk-card {cls}">'
                        f'<div style="margin-bottom:6px;">{tags}</div>'
                        f'{r["text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — BROWSE
# ════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.subheader("Browse Indexed Data")

    try:
        stats = api.stats(active_col)
        files = stats.get("files", [])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Chunks",  stats["total_chunks"])
        c2.metric("Files Indexed", len(files))
        c3.metric("Embedding",     f"{stats['provider']} · {stats['dimensions']}D")
        st.divider()

        if not files:
            st.info("No files indexed yet. Use the **Index Data** tab to add content.")
        else:
            st.markdown(f"**{len(files)} file(s) in `{active_col}`:**")
            for f in files:
                st.markdown(f"• `{f}`")

    except Exception as e:
        st.error(f"Could not fetch stats: {e}")

    # ── Raw chunk viewer ───────────────────────────────────────────────
    st.divider()
    st.subheader("Raw Chunk Viewer")
    st.caption("Run a broad search to browse raw content in the collection.")

    browse_query = st.text_input("Browse query", value="",
                                  placeholder="Leave blank or enter a topic to browse relevant chunks")
    browse_k     = st.slider("Chunks to show", 5, 50, 20)

    if st.button("🔎 Load Chunks", type="secondary"):
        q = browse_query.strip() or "data information content summary"
        try:
            chunks = api.search(active_col, q, browse_k, {})
            if not chunks:
                st.info("No chunks found.")
            else:
                for i, c in enumerate(chunks):
                    m = c["metadata"]
                    with st.expander(
                        f"[{i+1}] `{m.get('filename','?')}` · "
                        f"score {c['score']:.3f} · "
                        f"source: {m.get('source','—')}",
                        expanded=False,
                    ):
                        st.text(c["text"])
                        st.json(m, expanded=False)
        except Exception as e:
            st.error(f"Failed: {e}")


# ════════════════════════════════════════════════════════════════════════
# TAB 4 — MANAGE
# ════════════════════════════════════════════════════════════════════════
with tab_manage:
    st.subheader("Manage Index")

    # ── Delete a file ──────────────────────────────────────────────────
    st.markdown("#### Remove a file from the index")
    st.caption("All chunks from the selected file will be deleted. "
               "Use this before re-uploading an updated version.")
    try:
        mgmt_files = api.files(active_col)
    except Exception:
        mgmt_files = []

    if not mgmt_files:
        st.info("No files indexed yet.")
    else:
        del_file = st.selectbox("Select file to remove", ["— select —"] + mgmt_files)
        if st.button("🗑️ Remove from index", type="secondary",
                     disabled=(del_file == "— select —")):
            try:
                api.delete_file(active_col, del_file)
                st.success(f"✅ All chunks for `{del_file}` removed.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # ── Delete entire collection ───────────────────────────────────────
    st.markdown("#### ⚠️ Delete entire collection")
    st.warning(f"This will permanently delete **all data** in `{active_col}`. This cannot be undone.")
    confirm = st.text_input(f"Type `{active_col}` to confirm deletion")
    if st.button("💥 Delete Collection", type="primary",
                 disabled=(confirm != active_col)):
        try:
            api.delete_collection(active_col)
            st.success(f"Collection `{active_col}` deleted.")
            del st.session_state["active_col"]
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
