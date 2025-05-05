# app.py (Condensed logic integration)
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import pandas as pd
import traceback

# --- Import OMR processing module ---
try:
    import omr_processing as omr
    print("OMR processing module loaded successfully.")
    NUM_Q = omr.NUM_Q
    MAX_SCORE = omr.NUM_Q * omr.MARKS_PER_Q
except ImportError: st.error("Fatal Error: Could not import omr_processing.py."); st.stop()
except AttributeError as e: st.error(f"Fatal Error: Constant missing in omr_processing.py ({e})."); st.stop()
except Exception as e: st.error(f"Error importing OMR script: {e}"); st.stop()

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="OMR Evaluation System")
st.title("📄 OMR Sheet Evaluation System")
st.markdown("Upload student sheet & answer key for evaluation.")
st.markdown("---")
col1, col2 = st.columns(2)

# --- Session State ---
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
    st.session_state.score = 0.0
    st.session_state.result_image = None
    st.session_state.detailed_results = []
    st.session_state.correct_answers_list = None

# --- Inputs ---
with col1:
    st.subheader("1. Upload Student's OMR Sheet")
    uploaded_student_file = st.file_uploader("Choose student OMR image", type=["png", "jpg", "jpeg"], key="student_uploader")
    if uploaded_student_file:
        try: st.image(Image.open(uploaded_student_file), caption="Uploaded Student Sheet", use_column_width='always')
        except Exception as e: st.error(f"Error displaying student preview: {e}"); uploaded_student_file = None
with col2:
    st.subheader("2. Provide Answer Key")
    key_input_method = st.radio("Key Method:", ("Enter Answers Manually", "Upload Reference Answer Sheet"), key="key_method", horizontal=True)
    temp_correct_answers_list = None
    if key_input_method == "Enter Answers Manually":
        manual_answers = {}; options = ["A", "B", "C", "D"]; grid_cols = st.columns(min(NUM_Q, 4))
        for i in range(NUM_Q):
            col_idx = i % len(grid_cols)
            with grid_cols[col_idx]: manual_answers[f"Q{i+1}"] = st.selectbox(f"Q{i+1}:", options, key=f"q_{i}", index=None, placeholder="Select...")
        if all(manual_answers.values()):
            try: temp_correct_answers_list = [options.index(manual_answers[f"Q{i+1}"]) for i in range(NUM_Q)]
            except ValueError: st.error("Invalid option.")
    elif key_input_method == "Upload Reference Answer Sheet":
        uploaded_ref_file = st.file_uploader("Choose reference answer sheet image", type=["png", "jpg", "jpeg"], key="ref_uploader")
        if uploaded_ref_file:
            st.image(Image.open(uploaded_ref_file), caption="Uploaded Reference Sheet", use_column_width='always')
            with st.spinner("Processing reference sheet..."):
                try:
                    processed_ref_img = omr.load_and_preprocess_image(uploaded_ref_file.getvalue(), skip_alignment=False)
                    if processed_ref_img is None: st.error("Ref preprocess failed.")
                    else:
                        extracted_answers, _ = omr.get_answers_from_image(processed_ref_img)
                        if extracted_answers is None: st.error("Could not get answers from ref.")
                        elif -1 in extracted_answers: st.warning(f"Multiple marks on ref Q(s): {[i+1 for i, ans in enumerate(extracted_answers) if ans == -1]}. Cannot use.")
                        elif None in extracted_answers: st.warning(f"Unmarked Q(s) on ref: {[i+1 for i, ans in enumerate(extracted_answers) if ans is None]}. Cannot use.")
                        else: temp_correct_answers_list = extracted_answers; st.success(f"Ref answers: {[chr(65 + ans) for ans in temp_correct_answers_list]}")
                except Exception as e: st.error(f"Ref processing error: {e}")
    if temp_correct_answers_list is not None: st.session_state.correct_answers_list = temp_correct_answers_list
st.markdown("---")

# --- Evaluation Button & Logic ---
st.subheader("3. Evaluate")
button_disabled = not (uploaded_student_file and st.session_state.get('correct_answers_list') is not None)
evaluate_button = st.button("Evaluate Student Sheet", type="primary", use_container_width=True, disabled=button_disabled)
if button_disabled and (uploaded_student_file or st.session_state.get('correct_answers_list') is not None):
     missing = []
     if not uploaded_student_file: missing.append("Student Sheet")
     if not st.session_state.get('correct_answers_list'): missing.append("Answer Key")
     st.warning(f"⚠️ Please provide: {', '.join(missing)}")

if evaluate_button and not button_disabled:
    st.session_state.evaluation_done = False
    st.info("⏳ Starting evaluation...")
    # Add a placeholder for debug output if needed
    # debug_placeholder = st.empty()
    # debug_placeholder.write("--- DEBUG INFO START ---")

    processed_student_img, student_answers_indices, bubble_data = None, None, None # Initialize
    final_score, result_image, detailed_results_list = None, None, []

    try:
        processed_student_img = omr.load_and_preprocess_image(uploaded_student_file.getvalue(), skip_alignment=False)
        if processed_student_img is None: st.error("❌ Preprocessing failed.")
        else:
            # Get answers AND bubble data now
            student_answers_indices, bubble_data = omr.get_answers_from_image(processed_student_img)
            if student_answers_indices is None: st.error("❌ Could not extract student answers.")
            elif not bubble_data: st.error("❌ Failed to get bubble data for scoring.") # Check bubble data
            else:
                 # Call scoring with bubble_data
                 final_score, result_image, detailed_results_list = omr.score_student_sheet(
                     student_answers_indices,
                     st.session_state.correct_answers_list,
                     bubble_data, # Pass bubble data here
                     processed_student_img
                 )
                 if final_score is not None and result_image is not None:
                      st.session_state.score = final_score; st.session_state.result_image = result_image
                      st.session_state.detailed_results = detailed_results_list; st.session_state.evaluation_done = True
                 else: st.error("❌ Scoring failed to return results.")
    except Exception as e:
         st.error(f"❌ Evaluation Error: {e}"); st.code(traceback.format_exc())
    # finally: debug_placeholder.write("--- DEBUG INFO END ---") # Clear or update debug area


# --- Display Results ---
if st.session_state.get('evaluation_done', False):
    st.success("✅ Evaluation Complete!"); st.subheader("Results")
    st.metric(label="Final Score", value=f"{st.session_state.score:.2f} / {MAX_SCORE:.2f}")
    if st.session_state.result_image is not None:
        try:
             result_image_rgb = cv2.cvtColor(st.session_state.result_image, cv2.COLOR_BGR2RGB)
             st.image(result_image_rgb, caption="Evaluation Results on Student Sheet", use_column_width='always')
        except Exception as e: st.error(f"Error displaying result image: {e}"); st.code(traceback.format_exc())
    else: st.warning("Result image not available.")
    if st.session_state.detailed_results:
        st.subheader("Detailed Breakdown")
        try:
            df_results = pd.DataFrame(st.session_state.detailed_results, columns=["Q", "Marked", "Correct", "Status", "_intensities"])
            st.dataframe(df_results[["Q", "Marked", "Correct", "Status"]], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"Error displaying detailed results table: {e}")
    else: st.info("No detailed breakdown available.")
elif evaluate_button and not button_disabled:
     st.warning("Evaluation triggered, but results are not available. Check for errors above or in console.")