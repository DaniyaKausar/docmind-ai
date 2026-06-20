import streamlit as st
from dotenv import load_dotenv
import os
from rag.storage import (save_session, load_all_sessions, load_session, delete_session, save_analytics, load_analytics)
import uuid
from rag.loader import load_and_split_pdf
from rag.vectorstore import create_vectorstore, get_pdf_hash
from rag.chain import build_qa_chain, get_answer, generate_document_summary

load_dotenv()

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0b0f; }
section[data-testid="stSidebar"] { background-color: #0f1117 !important; border-right: 1px solid #1e2130; }
#MainMenu, footer, header { visibility: hidden; }
.hero-title { font-size: 2.8rem; font-weight: 700; letter-spacing: -0.03em; color: #ffffff; line-height: 1.1; margin-bottom: 4px; }
.hero-accent { color: #6366f1; }
.hero-sub { color: #4b5263; font-size: 0.95rem; font-weight: 400; margin-bottom: 24px; }
.user-bubble { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 12px 12px 4px 12px; padding: 14px 18px; margin: 8px 0; color: #e2e8f0; font-size: 0.95rem; }
.ai-bubble { background: #0f1117; border: 1px solid #1e2130; border-left: 3px solid #6366f1; border-radius: 4px 12px 12px 12px; padding: 14px 18px; margin: 8px 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.7; }
.conf-high { display: inline-block; background: #0d2818; color: #4ade80; border: 1px solid #166534; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.conf-medium { display: inline-block; background: #1c1a09; color: #fbbf24; border: 1px solid #854d0e; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.conf-low { display: inline-block; background: #1c0d0d; color: #f87171; border: 1px solid #991b1b; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.source-tag { display: inline-block; background: #13141f; color: #6366f1; border: 1px solid #2a2d4e; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin: 2px; }
.meta-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.summary-card { background: #0f1117; border: 1px solid #1e2130; border-top: 3px solid #6366f1; border-radius: 8px; padding: 16px 20px; margin: 12px 0; color: #c4c9d4; font-size: 0.88rem; line-height: 1.7; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }
.stat-box { background: #13141f; border: 1px solid #1e2130; border-radius: 8px; padding: 10px; text-align: center; }
.stat-num { font-size: 1.3rem; font-weight: 700; color: #6366f1; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 0.7rem; color: #4b5263; margin-top: 2px; }
.empty-state { text-align: center; padding: 60px 20px; color: #2a2d3e; }
.empty-icon { font-size: 4rem; margin-bottom: 16px; }
.empty-title { font-size: 1.4rem; font-weight: 600; color: #3a3d4e; margin-bottom: 8px; }
.empty-sub { font-size: 0.9rem; color: #2a2d3e; }
.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 40px; }
.feature-card { background: #0f1117; border: 1px solid #1a1d2e; border-radius: 10px; padding: 20px; text-align: left; }
.feature-icon { font-size: 1.5rem; margin-bottom: 10px; }
.feature-title { font-size: 0.9rem; font-weight: 600; color: #6366f1; margin-bottom: 6px; }
.feature-desc { font-size: 0.8rem; color: #4b5263; line-height: 1.5; }
.doc-label { background: #13141f; border: 1px solid #1e2130; border-left: 3px solid #6366f1; border-radius: 4px; padding: 8px 12px; font-size: 0.82rem; color: #6366f1; font-family: 'JetBrains Mono', monospace; word-break: break-all; margin: 8px 0; }
.stButton > button { background: #13141f !important; color: #a5b4fc !important; border: 1px solid #2a2d4e !important; border-radius: 6px !important; font-size: 0.8rem !important; padding: 6px 12px !important; }
.stButton > button:hover { background: #1a1d2e !important; border-color: #6366f1 !important; color: #ffffff !important; }
</style>""", unsafe_allow_html=True)

# Session state
defaults = {
    "chain_and_retriever": None, "chat_history": [], "pdf_name": None,
    "pdf_stats": {}, "doc_summary": None, "suggested_questions": [],
    "prefill_question": None, "session_id": str(uuid.uuid4()),
    "career_results": None, "career_resume_name": None,
    "multi_pdf_texts": {}, "multi_pdf_results": None,
    "multi_pdf_mode": None, "multi_pdf_names": [], "multi_pdf_stats": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
with st.sidebar:
    st.markdown("""<div style='padding: 4px 0 16px 0;'>
        <div style='font-size:1.2rem;font-weight:700;color:#fff;'>🧠 DocMind <span style='color:#6366f1'>AI</span></div>
        <div style='font-size:0.75rem;color:#4b5263;margin-top:2px'>Intelligent Document Analysis</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    all_sessions = load_all_sessions()
    if all_sessions:
        st.markdown("<div style='font-size:0.8rem;color:#6b7280;font-weight:600;letter-spacing:0.05em'>PAST SESSIONS</div>", unsafe_allow_html=True)
        for sid, sdata in list(reversed(list(all_sessions.items())))[:5]:
            col_s, col_d = st.columns([4, 1])
            with col_s:
                if st.button(f"📄 {sdata['pdf_name'][:20]}... ({sdata['message_count']} msgs)", key=f"sess_{sid}", use_container_width=True):
                    st.session_state.chat_history = sdata["chat_history"]
                    st.session_state.pdf_name = sdata["pdf_name"]
                    st.session_state.session_id = sid
                    st.info("✅ Session restored! Re-upload PDF to ask new questions.")
                    st.rerun()
            with col_d:
                if st.button("🗑", key=f"del_{sid}"):
                    delete_session(sid)
                    st.rerun()
        st.divider()

    st.markdown("<div style='font-size:0.8rem;color:#6b7280;font-weight:600;letter-spacing:0.05em'>DOCUMENT</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            progress = st.progress(0, text="📖 Reading PDF...")
            pdf_hash = get_pdf_hash(uploaded_file)
            chunks = load_and_split_pdf(uploaded_file)
            progress.progress(30, text="🔢 Creating embeddings...")
            vectorstore = create_vectorstore(chunks, pdf_hash=pdf_hash)
            progress.progress(60, text="🔗 Building AI chain...")
            st.session_state.chain_and_retriever = build_qa_chain(vectorstore, chunks)
            progress.progress(80, text="🧠 Analysing document...")
            summary = generate_document_summary(vectorstore)
            progress.progress(90, text="✅ Almost done...")
            st.session_state.doc_summary = summary
            progress.progress(90, text="✅ Almost done...")
            st.session_state.doc_summary = summary

            questions = []
            for line in summary.split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and "." in line[:3]:
                    q = line.split(".", 1)[-1].strip()
                    if q:
                        questions.append(q)
            st.session_state.suggested_questions = questions[:4]
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.pdf_stats = {
                "pages": len(chunks) // 3,
                "chunks": len(chunks),
                "size": f"{uploaded_file.size / 1024:.0f} KB"
            }
            save_analytics("pdf_processed", {"pdf_name": uploaded_file.name})
            progress.progress(100, text="✅ Ready!")

    if st.session_state.pdf_name:
        st.markdown(f"<div class='doc-label'>📄 {st.session_state.pdf_name}</div>", unsafe_allow_html=True)
        stats = st.session_state.pdf_stats
        st.markdown(f"""<div class='stat-grid'>
            <div class='stat-box'><div class='stat-num'>{stats.get('chunks', '-')}</div><div class='stat-label'>Chunks</div></div>
            <div class='stat-box'><div class='stat-num'>{stats.get('size', '-')}</div><div class='stat-label'>Size</div></div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        if st.session_state.suggested_questions:
            st.markdown("<div style='font-size:0.8rem;color:#6b7280;font-weight:600;letter-spacing:0.05em;margin-bottom:8px'>SUGGESTED</div>", unsafe_allow_html=True)
            for q in st.session_state.suggested_questions:
                if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
                    st.session_state.prefill_question = q
        st.divider()
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("<div style='font-size:0.72rem;color:#2a2d3e;text-align:center'>LangChain · Groq LLaMA 3.1 · ChromaDB<br>Sentence Transformers · Streamlit</div>", unsafe_allow_html=True)

# Main
st.markdown("""<div class='hero-title'>Doc<span class='hero-accent'>Mind</span> AI</div>
<div class='hero-sub'>Intelligent Document Analysis · Career Intelligence Platform</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💬 Document Chat", "🎯 Career Suite", "📚 Multi-PDF", "📊 Analytics"])

with tab1:
    if not st.session_state.chain_and_retriever:
        st.markdown("""<div class='empty-state'>
            <div class='empty-icon'>📄</div>
            <div class='empty-title'>No document loaded</div>
            <div class='empty-sub'>Upload a PDF from the sidebar to begin</div>
        </div>
        <div class='feature-grid'>
            <div class='feature-card'><div class='feature-icon'>🔍</div><div class='feature-title'>Semantic Search</div><div class='feature-desc'>Finds meaning, not just keywords.</div></div>
            <div class='feature-card'><div class='feature-icon'>🎯</div><div class='feature-title'>Confidence Scores</div><div class='feature-desc'>Every answer rated High/Medium/Low.</div></div>
            <div class='feature-card'><div class='feature-icon'>🧠</div><div class='feature-title'>Conversation Memory</div><div class='feature-desc'>Ask follow-up questions naturally.</div></div>
        </div>""", unsafe_allow_html=True)
    else:
        if st.session_state.doc_summary:
            with st.expander("📋 Document Briefing", expanded=False):
                st.markdown(f"<div class='summary-card'>{st.session_state.doc_summary}</div>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'>🙋 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                conf = msg.get("confidence", "Medium")
                conf_reason = msg.get("confidence_reason", "")
                sources = msg.get("sources", [])
                conf_class = f"conf-{conf.lower()}"
                source_tags = " ".join(f"<span class='source-tag'>{s}</span>" for s in sources)
                conf_title = f"{'✓' if conf == 'High' else '~' if conf == 'Medium' else '?'} {conf}"
                st.markdown(f"""<div class='ai-bubble'>{msg['content']}
                    <div class='meta-row'><span class='{conf_class}' title='{conf_reason}'>{conf_title}</span>{source_tags}</div>
                </div>""", unsafe_allow_html=True)

        prefill = st.session_state.pop("prefill_question", None)
        question = st.chat_input("Ask anything about your document...") or prefill

        if question:
            import time
            st.markdown(f"<div class='user-bubble'>🙋 {question}</div>", unsafe_allow_html=True)
            with st.spinner("Searching document..."):
                start_time = time.time()
                answer, sources, confidence, confidence_reason = get_answer(
                    st.session_state.chain_and_retriever,
                    question,
                    st.session_state.chat_history
                )
                response_time = round(time.time() - start_time, 2)
            conf_class = f"conf-{confidence.lower()}"
            source_tags = " ".join(f"<span class='source-tag'>{s}</span>" for s in sources)
            conf_title=f"{'✓'if confidence == 'High' else '∽' if confidence == 'Medium' else '?'} {confidence}"
            st.markdown(f"""
            <div class='ai-bubble'>
                {answer}
                <div class='meta-row'>
                    <span class='{conf_class}' title='{confidence_reason}'>{conf_title}</span>
                    {source_tags}
                    <span style='display:inline-block;background:#13141f;color:#4b5263;
                                border:1px solid #1e2130;border-radius:4px;padding:2px 8px;
                                font-size:0.72rem;font-family:monospace'>⚡ {response_time}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            conf_title = f"{'✓' if confidence == 'High' else '~' if confidence == 'Medium' else '?'} {confidence}"
            st.markdown(f"""<div class='ai-bubble'>{answer}
                <div class='meta-row'><span class='{conf_class}' title='{confidence_reason}'>{conf_title}</span>{source_tags}</div>
            </div>""", unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "sources": sources, "confidence": confidence, "confidence_reason": confidence_reason
            })
            save_session(st.session_state.session_id, st.session_state.pdf_name, st.session_state.chat_history)
            save_analytics("question_asked", {"confidence": confidence, "pdf": st.session_state.pdf_name, "response_time": response_time})

with tab2:
    from career.extractor import extract_resume_data, extract_jd_data
    from career.ats_engine import calculate_ats_score
    from career.gap_analyzer import find_missing_skills, generate_learning_roadmap, generate_resume_improvements
    from career.interview_prep import generate_interview_questions
    import pypdf

    st.markdown("### 🎯 Career Intelligence Suite")
    st.caption("Upload your resume + job description to get ATS score, skill gap analysis, and interview prep")

    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf", key="resume_upload")
    with col2:
        jd_option = st.radio("Job Description", ["Upload PDF", "Paste Text"], horizontal=True)
        if jd_option == "Upload PDF":
            jd_file = st.file_uploader("📋 Upload JD (PDF)", type="pdf", key="jd_upload")
            jd_text_input = None
        else:
            jd_file = None
            jd_text_input = st.text_area("Paste job description here...", height=150)

    analyze_btn = st.button("🚀 Analyze Now", type="primary", use_container_width=True)

    if analyze_btn:
        if not resume_file:
            st.error("Please upload your resume!")
        elif not jd_file and not jd_text_input:
            st.error("Please provide a job description!")
        else:
            with st.spinner("🧠 Analyzing... (30-60 seconds)"):
                def extract_pdf_text(file):
                    file.seek(0)
                    reader = pypdf.PdfReader(file)
                    return " ".join(page.extract_text() or "" for page in reader.pages)

                resume_text = extract_pdf_text(resume_file)
                jd_text = extract_pdf_text(jd_file) if jd_file else jd_text_input

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    resume_future = executor.submit(extract_resume_data, resume_text)
                    jd_future = executor.submit(extract_jd_data, jd_text)
                    resume_data = resume_future.result()
                    jd_data = jd_future.result()

                ats_scores = calculate_ats_score(resume_data, jd_data, resume_text)
                missing_req, missing_pref = find_missing_skills(resume_data, jd_data)

                st.session_state.career_results = {
                    "resume_data": resume_data, "jd_data": jd_data,
                    "ats_scores": ats_scores, "missing_req": missing_req, "missing_pref": missing_pref,
                }
                st.session_state.career_resume_name = resume_file.name
            st.success("✅ Analysis complete!")

    if st.session_state.career_results:
        res = st.session_state.career_results
        resume_data = res["resume_data"]
        jd_data = res["jd_data"]
        ats_scores = res["ats_scores"]
        missing_req = res["missing_req"]
        missing_pref = res["missing_pref"]

        st.info(f"📄 Showing results for: **{st.session_state.career_resume_name}**")
        st.markdown("---")
        st.markdown("## 📊 ATS Score Dashboard")

        score = ats_scores["overall_score"]
        color = "#4ade80" if score >= 75 else "#fbbf24" if score >= 55 else "#f87171"
        st.markdown(f"""<div style='text-align:center;padding:24px;background:#0f1117;border:1px solid #1e2130;border-radius:12px;margin:16px 0'>
            <div style='font-size:0.85rem;color:#6b7280;letter-spacing:0.1em'>OVERALL ATS SCORE</div>
            <div style='font-size:4rem;font-weight:800;color:{color};font-family:JetBrains Mono,monospace'>{score}</div>
            <div style='font-size:1rem;color:{color}'>{ats_scores["hiring_readiness"]}</div>
        </div>""", unsafe_allow_html=True)

        for label, val in [
            ("🎯 Skill Match", ats_scores["skill_match"]),
            ("🔑 Keyword Match", ats_scores["keyword_match"]),
            ("💼 Experience", ats_scores["experience_alignment"]),
            ("🎓 Education", ats_scores["education_alignment"]),
            ("🚀 Projects", ats_scores["project_relevance"]),
        ]:
            bar_color = "#4ade80" if val >= 75 else "#fbbf24" if val >= 55 else "#f87171"
            st.markdown(f"""<div style='margin:8px 0'>
                <div style='display:flex;justify-content:space-between;font-size:0.85rem;color:#c4c9d4;margin-bottom:4px'>
                    <span>{label}</span><span style='color:{bar_color};font-weight:600'>{val}%</span>
                </div>
                <div style='background:#1e2130;border-radius:4px;height:8px'>
                    <div style='background:{bar_color};width:{val}%;height:8px;border-radius:4px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### ✅ Matched Keywords")
            for kw in ats_scores["matched_keywords"]:
                st.markdown(f"<span class='source-tag'>✓ {kw}</span>", unsafe_allow_html=True)
        with col_b:
            st.markdown("#### ❌ Missing Keywords")
            for kw in ats_scores["missing_keywords"]:
                st.markdown(f"<span style='display:inline-block;background:#1c0d0d;color:#f87171;border:1px solid #991b1b;border-radius:4px;padding:2px 8px;font-size:0.72rem;font-family:monospace;margin:2px'>✗ {kw}</span>", unsafe_allow_html=True)

        ct1, ct2, ct3 = st.tabs(["🔍 Skill Gap", "📝 Resume Tips", "🎤 Interview Prep"])
        with ct1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔴 Missing Required")
                for s in missing_req: st.markdown(f"- {s}")
            with c2:
                st.markdown("#### 🟡 Missing Preferred")
                for s in missing_pref: st.markdown(f"- {s}")
            if st.button("Generate Learning Roadmap 🗺️"):
                with st.spinner("Creating roadmap..."):
                    roadmap = generate_learning_roadmap(resume_data, jd_data, missing_req, missing_pref)
                st.markdown(roadmap)
        with ct2:
            if st.button("Generate Resume Improvements ✍️"):
                with st.spinner("Analyzing resume..."):
                    improvements = generate_resume_improvements(resume_data, jd_data, ats_scores)
                st.markdown(improvements)
        with ct3:
            if st.button("Generate Interview Questions 🎤"):
                with st.spinner("Preparing questions..."):
                    iq = generate_interview_questions(resume_data, jd_data)
                st.markdown(iq)

with tab3:
    from rag.multi_pdf import (load_pdf_to_text, load_pdf_to_vectorstore,
        generate_pdf_summary, compare_two_pdfs, compare_multiple_pdfs, cross_pdf_query)
    os.makedirs("chroma_multi", exist_ok=True)

    st.markdown("### 📚 Multi-PDF Intelligence")
    st.caption("Upload multiple PDFs — summaries, comparisons, cross-document answers")

    uploaded_pdfs = st.file_uploader("Upload PDFs (up to 5)", type="pdf", accept_multiple_files=True, key="multi_pdf_upload")

    if uploaded_pdfs:
        if len(uploaded_pdfs) > 5:
            uploaded_pdfs = uploaded_pdfs[:5]
        st.success(f"✅ {len(uploaded_pdfs)} PDF(s): {', '.join(f.name for f in uploaded_pdfs)}")

    mode = st.radio("What do you want to do?", ["📋 Summarize All", "⚡ Compare Two", "🌐 Common Themes", "❓ Ask Across All"], horizontal=True)
    go_btn = st.button("🚀 Go!", use_container_width=True, type="primary")

    if go_btn and uploaded_pdfs:
        import concurrent.futures
        with st.spinner("Processing PDFs in parallel..."):
            def process_pdf(f):
                f.seek(0)
                text, pages = load_pdf_to_text(f)
                f.seek(0)
                vs, chunks, _ = load_pdf_to_vectorstore(f, f.name)
                return f.name, text, pages, vs

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_pdf, f) for f in uploaded_pdfs]
                results_raw = [fut.result() for fut in concurrent.futures.as_completed(futures)]

            pdf_texts = {r[0]: r[1] for r in results_raw}
            pdf_vectorstores = {r[0]: r[3] for r in results_raw}
            pdf_stats_data = {r[0]: r[2] for r in results_raw}

        with st.spinner("Running analysis..."):
            result_data = None
            if mode == "📋 Summarize All":
                def summarize(item):
                    name, text = item
                    return name, generate_pdf_summary(text, name)
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    summaries = dict(executor.map(summarize, pdf_texts.items()))
                result_data = {"mode": mode, "summaries": summaries}
            elif mode == "⚡ Compare Two":
                names = list(pdf_texts.keys())
                if len(names) >= 2:
                    comparison = compare_two_pdfs(pdf_texts[names[0]], names[0], pdf_texts[names[1]], names[1])
                    result_data = {"mode": mode, "comparison": comparison, "names": names}
                else:
                    st.error("Upload at least 2 PDFs!")
            elif mode == "🌐 Common Themes":
                themes = compare_multiple_pdfs(pdf_texts)
                result_data = {"mode": mode, "themes": themes}
            elif mode == "❓ Ask Across All":
                result_data = {"mode": mode, "vectorstores": pdf_vectorstores, "pdf_texts": pdf_texts}

        if result_data:
            st.session_state.multi_pdf_results = result_data
            st.session_state.multi_pdf_names = list(pdf_texts.keys())
            st.session_state.multi_pdf_mode = mode
            st.session_state.multi_pdf_stats = pdf_stats_data

    if st.session_state.multi_pdf_results:
        res = st.session_state.multi_pdf_results
        mode_used = res["mode"]
        st.markdown("---")
        st.info(f"📄 Loaded: **{', '.join(st.session_state.multi_pdf_names)}**")

        if st.session_state.multi_pdf_stats:
            cols = st.columns(max(1, len(st.session_state.multi_pdf_names)))
            for i, name in enumerate(st.session_state.multi_pdf_names):
                pages = st.session_state.multi_pdf_stats.get(name, "?")
                with cols[i]:
                    st.markdown(f"<div class='stat-box'><div class='stat-num'>{pages}</div><div class='stat-label'>{name[:12]}...</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        if mode_used == "📋 Summarize All":
            st.markdown("## 📋 Individual Summaries")
            for pdf_name, summary in res["summaries"].items():
                with st.expander(f"📄 {pdf_name}", expanded=True):
                    st.markdown(f"<div class='summary-card'>{summary}</div>", unsafe_allow_html=True)
        elif mode_used == "⚡ Compare Two":
            names = res.get("names", [])
            if len(names) >= 2:
                st.markdown(f"## ⚡ {names[0]} vs {names[1]}")
                st.markdown(f"<div class='summary-card'>{res['comparison']}</div>", unsafe_allow_html=True)
        elif mode_used == "🌐 Common Themes":
            st.markdown("## 🌐 Common Themes")
            st.markdown(f"<div class='summary-card'>{res['themes']}</div>", unsafe_allow_html=True)
        elif mode_used == "❓ Ask Across All":
            st.markdown("## ❓ Ask Across All PDFs")
            cross_question = st.text_input("Ask a question...", key="cross_q", placeholder="What are the main differences?")
            if st.button("Search All PDFs 🔍"):
                with st.spinner("Searching across all documents..."):
                    cross_results = cross_pdf_query(res["vectorstores"], cross_question)
                for pdf_name, result in cross_results.items():
                    with st.expander(f"📄 {pdf_name}", expanded=True):
                        source_tags = "".join(f"<span class='source-tag'>{p}</span>" for p in result['pages'])
                        st.markdown(f"<div class='ai-bubble'>{result['answer']}<div class='meta-row'>{source_tags}</div></div>", unsafe_allow_html=True)

        st.divider()
        if st.button("🗑 Clear Multi-PDF Results", use_container_width=True):
            st.session_state.multi_pdf_results = None
            st.session_state.multi_pdf_names = []
            st.rerun()

with tab4:
    st.markdown("### 📊 Usage Analytics")
    analytics = load_analytics()
    totals = analytics.get("totals", {})
    events = analytics.get("events", [])

    if not totals or totals.get("questions_asked", 0) == 0:
        st.info("No analytics yet — start chatting to see your stats!")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total_q = totals.get("questions_asked", 1)
        high = totals.get("high_confidence", 0)
        med = totals.get("medium_confidence", 0)
        low = totals.get("low_confidence", 0)
        avg_conf = "High" if high > med and high > low else "Medium" if med >= low else "Low"
        conf_color = "#4ade80" if avg_conf == "High" else "#fbbf24" if avg_conf == "Medium" else "#f87171"

        for col, label, display, col_color in [
            (c1, "📄 PDFs Processed", totals.get("pdfs_processed", 0), "#6366f1"),
            (c2, "❓ Questions Asked", totals.get("questions_asked", 0), "#4ade80"),
            (c3, "🎯 Avg Confidence", avg_conf, conf_color),
            (c4, "💬 Sessions", totals.get("sessions", 0), "#60a5fa"),
        ]:
            with col:
                st.markdown(f"""<div class='stat-box' style='padding:16px'>
                    <div style='font-size:0.75rem;color:#6b7280'>{label}</div>
                    <div style='font-size:1.8rem;font-weight:700;color:{col_color};font-family:monospace'>{display}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🎯 Answer Confidence Breakdown")
        for label, count, color in [("✓ High", high, "#4ade80"), ("~ Medium", med, "#fbbf24"), ("? Low", low, "#f87171")]:
            pct = round((count / total_q) * 100) if total_q > 0 else 0
            st.markdown(f"""<div style='margin:8px 0'>
                <div style='display:flex;justify-content:space-between;font-size:0.85rem;color:#c4c9d4;margin-bottom:4px'>
                    <span>{label}</span><span style='color:{color}'>{count} ({pct}%)</span>
                </div>
                <div style='background:#1e2130;border-radius:4px;height:10px'>
                    <div style='background:{color};width:{pct}%;height:10px;border-radius:4px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🕐 Recent Activity")
        for event in list(reversed(events))[:10]:
            icon = "📄" if event["type"] == "pdf_processed" else "❓"
            label = (f"Processed: {event['data'].get('pdf_name', '')}" if event["type"] == "pdf_processed"
                     else f"Question asked — {event['data'].get('confidence','?')} confidence")
            st.markdown(f"""<div style='display:flex;justify-content:space-between;padding:8px 12px;
                background:#0f1117;border:1px solid #1e2130;border-radius:6px;margin:4px 0;font-size:0.82rem;color:#c4c9d4'>
                <span>{icon} {label}</span><span style='color:#4b5263'>{event['timestamp']}</span>
            </div>""", unsafe_allow_html=True)
