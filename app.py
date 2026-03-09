"""
Math Mentor — JEE Math Solver
Multimodal: Image (Gemini Vision) + Audio (Whisper) + Text
5 Agents: Parser → Router → Solver (RAG) → Verifier → Explainer
"""
import streamlit as st
from memory.memory_store import get_stats, save_feedback, find_similar_problems

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Math Mentor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.subtitle { color: #666; font-size: 0.9rem; margin-top: -10px; }
.answer-box { background: #f0f7ff; border-left: 4px solid #2196F3;
              padding: 16px; border-radius: 8px; margin: 10px 0; }
.agent-badge { display: inline-block; background: #e8f5e9; color: #2e7d32;
               border-radius: 12px; padding: 2px 10px; font-size: 0.8rem; margin: 2px; }
.hitl-box { background: #fff3e0; border-left: 4px solid #ff9800;
            padding: 12px; border-radius: 8px; }
.conf-high { color: #2e7d32; font-weight: bold; }
.conf-med  { color: #f57c00; font-weight: bold; }
.conf-low  { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    force_hitl = st.toggle("Force Human Review", value=False)
    st.divider()

    # Stats
    st.markdown("## 📊 Session Stats")
    stats = get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Total Solved", stats["total_solved"])
    col2.metric("With Feedback", stats["with_feedback"])

    if stats["topics"]:
        import pandas as pd
        topic_df = pd.DataFrame(
            list(stats["topics"].items()), columns=["Topic", "Count"]
        )
        st.bar_chart(topic_df.set_index("Topic"))

    st.divider()
    st.markdown("## 🔧 Stack")
    st.markdown("🖼️ **OCR:** Gemini Vision API")
    st.markdown("🤖 **LLM:** Groq LLaMA-3.1-8B")
    st.markdown("📚 **RAG:** FAISS + sentence-transformers")
    st.markdown("🎙️ **ASR:** Whisper (local)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧮 Math Mentor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">JEE-Style Math Solver · RAG + Multi-Agent + HITL + Memory · '
    'Gemini Vision + Groq LLaMA</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Input Mode ────────────────────────────────────────────────────────────────
st.markdown("### 📥 Input Mode")
input_mode = st.radio(
    "Select input type:",
    ["✍️ Text", "🖼️ Image (OCR)", "🎙️ Audio (Speech)"],
    horizontal=True,
    label_visibility="collapsed",
)

raw_input = ""
input_source = "text"
ocr_confidence = 1.0
asr_confidence = 1.0

# Text Input
if input_mode == "✍️ Text":
    input_source = "text"
    raw_input = st.text_area(
        "Type your math problem:",
        placeholder="e.g. Find the 20th term of AP: 3, 7, 11, ..., 107\n(A) 79  (B) 83  (C) 75  (D) 87",
        height=120,
    )

# Image Input
elif input_mode == "🖼️ Image (OCR)":
    input_source = "ocr"
    uploaded_img = st.file_uploader(
        "Upload image of math problem (JPG/PNG):",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_img:
        col_img, col_text = st.columns([1, 1])
        with col_img:
            st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        with col_text:
            with st.spinner("🤖 Gemini Vision reading your image..."):
                try:
                    from input_handlers.ocr_handler import extract_text_from_image, preprocess_math_text
                    img_bytes = uploaded_img.read()
                    extracted, ocr_confidence = extract_text_from_image(img_bytes)
                    extracted = preprocess_math_text(extracted)
                except Exception as e:
                    extracted = f"[Vision Error: {e}]"
                    ocr_confidence = 0.0

            if ocr_confidence >= 0.80:
                st.success(f"✅ Gemini Vision extracted text ({ocr_confidence:.0%} confidence)")
            elif ocr_confidence >= 0.55:
                st.warning(f"⚠️ Review extracted text ({ocr_confidence:.0%} confidence)")
            else:
                st.error("❌ Extraction failed — please type the problem manually")

            raw_input = st.text_area(
                "Extracted Text (edit if needed):",
                value=extracted,
                height=150,
                help="Gemini Vision read your image. Edit if anything looks wrong."
            )

# Audio Input
elif input_mode == "🎙️ Audio (Speech)":
    input_source = "audio"
    uploaded_audio = st.file_uploader(
        "Upload audio file (WAV/MP3/M4A):",
        type=["wav", "mp3", "m4a", "ogg"]
    )
    if uploaded_audio:
        st.audio(uploaded_audio)
        with st.spinner("🎙️ Whisper transcribing audio..."):
            try:
                from input_handlers.audio_handler import transcribe_audio, normalize_math_speech
                ext = uploaded_audio.name.split(".")[-1]
                transcript, asr_confidence = transcribe_audio(uploaded_audio.read(), ext)
                transcript = normalize_math_speech(transcript)
            except Exception as e:
                transcript = f"[ASR Error: {e}]"
                asr_confidence = 0.0

        if asr_confidence >= 0.70:
            st.success(f"✅ Transcribed ({asr_confidence:.0%} confidence)")
        elif asr_confidence >= 0.50:
            st.warning(f"⚠️ Review transcript ({asr_confidence:.0%} confidence)")
        else:
            st.error("❌ Low confidence — please review")

        raw_input = st.text_area(
            "Transcript (edit if needed):",
            value=transcript,
            height=120,
        )

st.divider()

# ── Solve Button ──────────────────────────────────────────────────────────────
solve_btn = st.button("🚀 Solve Problem", type="primary", disabled=not raw_input.strip())

if solve_btn and raw_input.strip():
    # Check for similar past problems first
    similar = find_similar_problems(raw_input)
    if similar:
        st.info(f"🧠 Found {len(similar)} similar problem(s) in memory — see Memory tab below")

    with st.spinner("Running 5-agent pipeline..."):
        from pipeline import run_pipeline
        result = run_pipeline(
            raw_input=raw_input,
            input_source=input_source,
            ocr_confidence=ocr_confidence,
            asr_confidence=asr_confidence,
            force_hitl=force_hitl,
        )
    st.session_state["result"] = result
    st.session_state["problem_id"] = result.get("problem_id")

# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    parsed = result["parsed_problem"]
    solver = result["solver_result"]
    verifier = result["verifier_result"]
    explainer = result["explainer_result"]
    routing = result["routing"]

    # HITL Banner
    if result.get("hitl_triggered"):
        st.markdown(f"""
        <div class="hitl-box">
        🔍 <strong>Human Review Requested</strong><br>
        {result.get('hitl_reason', 'Please review before finalising')}
        </div>
        """, unsafe_allow_html=True)
        with st.expander("✏️ Edit Problem Before Solving", expanded=True):
            corrected = st.text_area(
                "Correct the problem statement:",
                value=parsed.get("problem_text", raw_input),
                height=100,
            )
            notes = st.text_input("Your notes / corrections:")
            if st.button("✅ Approve & Continue"):
                st.success("Approved! Scroll down for results.")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Answer", "📖 Explanation", "🔍 Agent Trace", "📚 Sources", "🧠 Memory"
    ])

    # Tab 1: Answer
    with tab1:
        st.markdown(f"**Problem:** {parsed.get('problem_text', raw_input)}")

        conf = verifier.get("confidence", 0)
        conf_pct = int(conf * 100)
        conf_class = "conf-high" if conf >= 0.80 else ("conf-med" if conf >= 0.60 else "conf-low")

        final_ans = verifier.get("verified_answer") or solver.get("final_answer", "")
        st.markdown(f"""
        <div class="answer-box">
        <h3>🎯 Final Answer</h3>
        <h2>{final_ans}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Confidence:** <span class='{conf_class}'>{conf_pct}%</span>", unsafe_allow_html=True)
        st.markdown(f"**Topic:** `{parsed.get('topic')}` / `{parsed.get('subtopic')}`")
        st.markdown(f"**Difficulty:** `{routing.get('difficulty')}`")
        st.markdown(f"**Strategy:** `{routing.get('strategy')}`")

        if verifier.get("is_correct"):
            st.success(f"✅ {verifier.get('feedback', 'Solution verified.')}")
        else:
            st.warning(f"⚠️ {verifier.get('feedback', 'Please review.')}")
            if verifier.get("issues"):
                for issue in verifier["issues"]:
                    st.write(f"• {issue}")

        st.markdown("---")
        st.markdown("**📝 Full Solution:**")
        st.markdown(solver.get("solution_text", ""))

        if solver.get("python_results"):
            with st.expander("🐍 Python Calculator Results"):
                for pr in solver["python_results"]:
                    if pr.get("success"):
                        st.code(f"# Code\n{pr['code']}\n\n# Result\n{pr['result']}", language="python")

        # Feedback
        st.divider()
        st.markdown("**Was this answer correct?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correct"):
                save_feedback(st.session_state.get("problem_id", ""), True)
                st.success("Thanks for the feedback!")
        with col2:
            comment = st.text_input("Comment (optional):", key="fb_comment")
            if st.button("❌ Incorrect"):
                save_feedback(st.session_state.get("problem_id", ""), False, comment)
                st.error("Feedback saved. We'll improve!")

    # Tab 2: Explanation
    with tab2:
        st.markdown("### 📖 Step-by-Step Explanation")
        st.markdown(explainer.get("explanation", "No explanation available."))
        st.info(f"**Concept used:** {explainer.get('concept_used', '')}")

    # Tab 3: Agent Trace
    with tab3:
        st.markdown("### 🔍 Agent Pipeline Trace")
        for step in result.get("agent_trace", []):
            status_icon = "✅" if step["status"] == "done" else "⏳"
            with st.expander(f"{status_icon} {step['agent']} Agent"):
                if "output" in step:
                    for k, v in step["output"].items():
                        st.write(f"**{k}:** {v}")

    # Tab 4: Sources
    with tab4:
        st.markdown("### 📚 Retrieved Sources (RAG)")
        sources = solver.get("retrieved_sources", [])
        if sources:
            for s in sources:
                st.markdown(
                    f'<span class="agent-badge">📄 {s["source"]} — score: {s["score"]:.3f}</span>',
                    unsafe_allow_html=True,
                )
            st.divider()
            chunks = solver.get("retrieved_chunks", [])
            for i, c in enumerate(chunks, 1):
                with st.expander(f"Chunk {i}: {c['source']}"):
                    st.text(c["text"])
        else:
            st.info("No sources retrieved.")

    # Tab 5: Memory
    with tab5:
        st.markdown("### 🧠 Memory & Similar Problems")
        similar = result.get("similar_problems", [])
        if similar:
            st.success(f"Found {len(similar)} similar problem(s) in memory:")
            for s in similar:
                with st.expander(f"[{s['topic']}] {s['problem_text'][:80]}... (similarity: {s.get('similarity', 0):.0%})"):
                    st.write(f"**Answer:** {s.get('final_answer', 'N/A')}")
                    st.write(f"**Confidence:** {s.get('confidence', 0):.0%}")
                    st.write(f"**Solved on:** {s.get('timestamp', '')[:10]}")
        else:
            st.info("No similar problems found in memory yet.")
