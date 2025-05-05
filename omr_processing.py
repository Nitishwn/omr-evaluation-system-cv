import cv2
import numpy as np
import logging
import os
import io
import csv
from PIL import Image


logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

STD_W, STD_H = 457, 683
BUBBLE_W, BUBBLE_H = 43, 43
BUBBLE_RADIUS = BUBBLE_W // 2
RING_WIDTH = 12

START_X, START_Y = 195, 153
SPACING_X, SPACING_Y = 67, 88

NUM_Q, NUM_O = 6, 4
MARKS_PER_Q = 1.0
NEG_MARK = 0.25

PARTIAL_PENALTY = 0.125 
FILL_THRESHOLD = 0.3
PARTIAL_THRESHOLD = 0.15 # Used to determine if a *wrong* answer gets partial penalty reduction
INCOMPLETE_THRESHOLD = 0.8 # Threshold for a 'perfect' correct mark
RING_THRESHOLD = 0.2      # Threshold for a 'perfect' correct mark (must be low) and for partial *wrong* ring penalty
CORRECT_ANSWER_SCORE = MARKS_PER_Q
WRONG_ANSWER_SCORE = -NEG_MARK
UNMARKED_ANSWER_SCORE = 0.0
FALLBACK_CORNERS = np.array([[50, 50], [STD_W - 50, 50], [STD_W - 50, STD_H - 50], [50, STD_H - 50]], dtype="float32")
OUTPUT_DIR = "output_steps"
try: os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError as e: logging.error(f"Could not create output directory '{OUTPUT_DIR}': {e}"); OUTPUT_DIR = None


def save_step_image(image, step_name):
    if not OUTPUT_DIR: return
    filepath = os.path.join(OUTPUT_DIR, f"step_{step_name}.png")
    try:
        if image is not None and image.size > 0:
            img_to_save = image
            if len(image.shape) == 3 and image.shape[2] == 4: img_to_save = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if len(img_to_save.shape) == 2 or (len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3): cv2.imwrite(filepath, img_to_save)
            else: logging.warning(f"Skipping save for {step_name}, unexpected shape {image.shape}.")
    except Exception as e: print(f"Error saving image {filepath}: {e}")

def order_points(pts):
    if pts is None: return None
    if pts.shape == (4, 1, 2): pts = pts.reshape(4, 2)
    if pts.shape != (4, 2): logging.error(f"Invalid shape for order_points: {pts.shape}"); return None
    rect = np.zeros((4, 2), dtype="float32"); s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]; rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def get_perspective_transform(image, corners):
    if corners is None: return None, None
    if corners.shape == (4,1,2): corners = corners.reshape(4,2)
    if corners.shape != (4, 2): return None, None
    rect = order_points(corners);
    if rect is None: return None, None
    (tl, tr, br, bl) = rect; widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl); max_width = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl); max_height = max(int(heightA), int(heightB))
    if max_width <= 0 or max_height <= 0: return None, None
    dst = np.array([[0, 0],[max_width - 1, 0],[max_width - 1, max_height - 1],[0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst); warped = cv2.warpPerspective(image, M, (max_width, max_height))
    save_step_image(warped, "06_warped_aligned_original_size"); return warped, M

def detect_border_corners(image, debug=True):
    if image is None or image.size == 0: return FALLBACK_CORNERS.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); save_step_image(gray, "01_grayscale")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0); save_step_image(blurred, "02_blurred")
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); save_step_image(thresh, "03_thresholded")
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return FALLBACK_CORNERS.copy()
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    found_rectangle = None; min_area_threshold = image.shape[0] * image.shape[1] * 0.01 # Ref threshold
    for contour in contours:
        area = cv2.contourArea(contour);
        if area < min_area_threshold: continue
        peri = cv2.arcLength(contour, True); approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4: found_rectangle = approx; break
    if found_rectangle is not None:
        corners = found_rectangle.reshape(4, 2).astype("float32"); ordered_corners = order_points(corners)
        if ordered_corners is None: return FALLBACK_CORNERS.copy()
        if debug:
            debug_image = image.copy(); cv2.drawContours(debug_image, [found_rectangle], -1, (0, 255, 0), 3)
            save_step_image(debug_image, "05_debug_contours")
        return ordered_corners
    else:
        if debug: save_step_image(image, "05_debug_contours_FAILED")
        return FALLBACK_CORNERS.copy()

def export_to_csv(bubble_data, filename="bubble_fill_levels.csv"):
    if not OUTPUT_DIR or not bubble_data: return
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file); writer.writerow(["Question", "Option", "Fill_Level", "Ring_Fill_Level", "X_Coord", "Y_Coord"])
            for row in bubble_data: writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", row[4], row[5]])
    except Exception as e: print(f"Error exporting CSV {filepath}: {e}")

def evaluate_bubbles(image_to_evaluate):
    """Calculates fill intensity using reference logic."""
    if image_to_evaluate is None: return [], None
    if len(image_to_evaluate.shape) == 3: gray = cv2.cvtColor(image_to_evaluate, cv2.COLOR_BGR2GRAY); bubble_viz_image = image_to_evaluate.copy()
    else: gray = image_to_evaluate.copy(); bubble_viz_image = cv2.cvtColor(image_to_evaluate, cv2.COLOR_GRAY2BGR)
    eval_h, eval_w = gray.shape[:2]; bubble_data = []; marked_answers = []; outer_radius = BUBBLE_RADIUS + RING_WIDTH
    for q in range(NUM_Q):
        intensities = []; ring_intensities = []
        for o in range(NUM_O):
            x = int(START_X + o * SPACING_X); y = int(START_Y + q * SPACING_Y); option_char = chr(65 + o); norm_intensity = 0.0; ring_norm_intensity = 0.0
            if (0 <= y < eval_h and 0 <= x < eval_w):
                cv2.circle(bubble_viz_image, (x, y), BUBBLE_RADIUS, (0, 255, 0), 1) # Viz circle
                inner_mask = np.zeros_like(gray); cv2.circle(inner_mask, (x, y), BUBBLE_RADIUS, 255, -1); mask_area = cv2.countNonZero(inner_mask)
                if mask_area > 0: mean_intensity = cv2.mean(gray, mask=inner_mask)[0]; norm_intensity = max(0.0, 1.0 - (mean_intensity / 255.0))
                outer_mask = np.zeros_like(gray); cv2.circle(outer_mask, (x, y), outer_radius, 255, -1); ring_mask = cv2.subtract(outer_mask, inner_mask); ring_mask_area = cv2.countNonZero(ring_mask)
                if ring_mask_area > 0: ring_mean_intensity = cv2.mean(gray, mask=ring_mask)[0]; ring_norm_intensity = max(0.0, 1.0 - (ring_mean_intensity / 255.0))
            intensities.append(norm_intensity); ring_intensities.append(ring_norm_intensity)
            bubble_data.append((q + 1, option_char, norm_intensity, ring_norm_intensity, x, y))
        marked_indices = [i for i, intensity in enumerate(intensities) if intensity >= FILL_THRESHOLD]
        marked_answers.append(marked_indices[0] if len(marked_indices) == 1 else (-1 if len(marked_indices) > 1 else None))
    save_step_image(bubble_viz_image, "07_bubble_locations")
    export_to_csv(bubble_data)
    return bubble_data, marked_answers

def generate_heatmap(heatmap_base_image, bubble_data):
    """Generates heatmap."""
    if heatmap_base_image is None or not bubble_data: return
    heatmap = cv2.cvtColor(heatmap_base_image, cv2.COLOR_GRAY2BGR) if len(heatmap_base_image.shape) == 2 else heatmap_base_image.copy()
    for _, _, intensity, ring_intensity, x, y in bubble_data:
        try: ix, iy = int(x), int(y)
        except ValueError: continue
        if not (0 <= iy < heatmap.shape[0] and 0 <= ix < heatmap.shape[1]): continue
        if intensity >= INCOMPLETE_THRESHOLD: color, alpha = (0, 255, 0), 0.6
        elif intensity >= PARTIAL_THRESHOLD: color, alpha = (0, 255, 255), 0.5
        else: color, alpha = (0, 0, 255), 0.4
        overlay = heatmap.copy(); cv2.circle(overlay, (ix, iy), BUBBLE_RADIUS, color, -1); cv2.addWeighted(overlay, alpha, heatmap, 1 - alpha, 0, heatmap)
        if intensity >= PARTIAL_THRESHOLD: cv2.circle(heatmap, (ix, iy), BUBBLE_RADIUS, (50, 50, 50), 1)
        if ring_intensity >= RING_THRESHOLD: cv2.circle(heatmap, (ix, iy), BUBBLE_RADIUS + RING_WIDTH, (0, 215, 255), 2)
    save_step_image(heatmap, "09_fill_heatmap")

# --- Main Processing Functions ---
# These represent the core steps in the OMR flow

def load_and_preprocess_image(image_bytes=None, image_path=None, skip_alignment=False):
    # (Function remains unchanged)
    image_cv = None
    try:
        if image_path: image_cv = cv2.imread(image_path)
        elif image_bytes: image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB'); image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        else: raise ValueError("No image source provided.")
        if image_cv is None: raise ValueError("Image loading returned None.")
        save_step_image(image_cv, "00_original_image")
    except Exception as e: logging.error(f"Image loading failed: {e}"); return None
    image_to_evaluate = None; interp_method = cv2.INTER_AREA
    if skip_alignment:
        aligned_resized = cv2.resize(image_cv, (STD_W, STD_H), interpolation=interp_method); save_step_image(aligned_resized, "06a_aligned_resized_skipped"); image_to_evaluate = aligned_resized
    else:
        corners = detect_border_corners(image_cv, debug=True)
        is_fallback = np.array_equal(corners, FALLBACK_CORNERS)
        if corners is not None and not is_fallback:
            aligned_image, _ = get_perspective_transform(image_cv, corners)
            if aligned_image is not None and aligned_image.size > 0: aligned_resized = cv2.resize(aligned_image, (STD_W, STD_H), interpolation=interp_method); save_step_image(aligned_resized, "06a_aligned_resized"); image_to_evaluate = aligned_resized
            else: aligned_resized = cv2.resize(image_cv, (STD_W, STD_H), interpolation=interp_method); save_step_image(aligned_resized, "06a_aligned_resized_from_original_post_warp_fail"); save_step_image(image_cv, "06_warped_aligned_FAILED"); image_to_evaluate = aligned_resized
        else: aligned_resized = cv2.resize(image_cv, (STD_W, STD_H), interpolation=interp_method); save_step_image(aligned_resized, "06a_aligned_resized_fallback"); image_to_evaluate = aligned_resized
    if image_to_evaluate is None or image_to_evaluate.size == 0: logging.error("Preprocessing failed."); return None
    return image_to_evaluate

def get_answers_from_image(processed_image):
    if processed_image is None: return None, []
    bubble_data, marked_answers_list = evaluate_bubbles(processed_image)
    return marked_answers_list, bubble_data

def score_student_sheet(student_answers_indices, correct_answers_indices, bubble_data_for_drawing, processed_student_image):
    """Scores sheet, draws results (NO Partial Correct Marks)."""
    score = 0.0; detailed_results = []
    if processed_student_image is None: evaluation_viz_image = np.zeros((STD_H, STD_W, 3), dtype=np.uint8)
    elif len(processed_student_image.shape) == 2: evaluation_viz_image = cv2.cvtColor(processed_student_image, cv2.COLOR_GRAY2BGR)
    else: evaluation_viz_image = processed_student_image.copy() # Assume BGR

    if not bubble_data_for_drawing: return 0.0, evaluation_viz_image, []
    bubble_data_by_q = {q_num: {} for q_num in range(1, NUM_Q + 1)} # Pre-initialize dict
    for q_num, option, intensity, ring_intensity, x, y in bubble_data_for_drawing:
        bubble_data_by_q[q_num][option] = {'intensity': intensity, 'ring_intensity': ring_intensity, 'x': int(x), 'y': int(y)}

    # --- Scoring Loop ---
    for q in range(NUM_Q):
        q_num = q + 1; marked_idx = student_answers_indices[q] if q < len(student_answers_indices) else None
        correct_idx = correct_answers_indices[q] if q < len(correct_answers_indices) else None
        question_data = bubble_data_by_q.get(q_num, {}); intensities = [question_data.get(chr(65+i), {'intensity':0.0})['intensity'] for i in range(NUM_O)]
        ring_intensities = [question_data.get(chr(65+i), {'ring_intensity':0.0})['ring_intensity'] for i in range(NUM_O)]; coords = [(question_data.get(chr(65+i), {'x':-1,'y':-1})['x'], question_data.get(chr(65+i), {'y':-1})['y']) for i in range(NUM_O)]
        current_question_status = "Error"; marked_option_char = "N/A"; correct_option_char = 'N/A'
        correct_coord = (-1, -1); marked_coord = (-1,-1)
        marked_intensity = 0.0; marked_ring_intensity = 0.0
        is_correct = False
        current_score_change = 0.0

        # Determine basic status and involved coordinates/intensities
        if correct_idx is None: status = "No Key Provided"; marked_char = chr(65 + marked_idx) if marked_idx not in [None, -1] else ('Multiple' if marked_idx == -1 else 'None'); current_question_status = status
        else:
            correct_option_char = chr(65 + correct_idx); correct_coord = coords[correct_idx]
            if marked_idx == -1: current_score_change = WRONG_ANSWER_SCORE; current_question_status = "Wrong (Multiple)"; marked_option_char = "Multiple"
            elif marked_idx is None: current_score_change = UNMARKED_ANSWER_SCORE; current_question_status = "Unmarked"; marked_option_char = "None"
            else: # Single Mark
                marked_option_char = chr(65 + marked_idx); marked_coord = coords[marked_idx]
                if marked_coord == (-1,-1): current_question_status = "Error (Coord Missing)"; current_score_change = WRONG_ANSWER_SCORE
                else:
                    marked_intensity = intensities[marked_idx]; marked_ring_intensity = ring_intensities[marked_idx]
                    is_correct = (marked_idx == correct_idx)
                    # --- MODIFIED Scoring Logic: NO PARTIAL CORRECT ---
                    if is_correct:
                        # Check for 'perfect' mark
                        if marked_intensity >= INCOMPLETE_THRESHOLD and marked_ring_intensity < RING_THRESHOLD:
                            current_score_change = CORRECT_ANSWER_SCORE; current_question_status = "Correct" # Simplified status
                        else:
                            # Any other correct mark (messy, incomplete, low fill) is now considered WRONG
                            current_score_change = WRONG_ANSWER_SCORE; current_question_status = "Wrong (Imperfect Mark)" # New status
                    else: # Incorrect Answer (logic for partial penalty on WRONG answers remains)
                        current_score_change = WRONG_ANSWER_SCORE
                        if marked_intensity >= INCOMPLETE_THRESHOLD: current_question_status = "Wrong (Full Mark)"
                        elif marked_intensity >= PARTIAL_THRESHOLD: current_score_change += PARTIAL_PENALTY; current_question_status = "Partially Wrong (Incomplete)"
                        elif marked_ring_intensity >= RING_THRESHOLD: current_score_change += PARTIAL_PENALTY; current_question_status = "Partially Wrong (Ring)"
                        else: current_question_status = "Wrong (Low Fill Confidence)"

        score += current_score_change

        # --- Simplified Drawing Logic ---
        # Draw outline around correct answer (if applicable)
        if correct_coord != (-1,-1):
            if marked_idx is None: cv2.circle(evaluation_viz_image, correct_coord, BUBBLE_RADIUS + 5, (0, 255, 255), 2) # Yellow if unmarked
            elif not is_correct or marked_idx == -1: cv2.circle(evaluation_viz_image, correct_coord, BUBBLE_RADIUS + 5, (0, 255, 0), 2) # Green if wrong/multiple

        # Draw on marked bubbles
        if marked_idx == -1: # Multiple marks
            for i in range(NUM_O):
                if intensities[i] >= FILL_THRESHOLD and coords[i] != (-1,-1): mx, my = coords[i]; cv2.line(evaluation_viz_image, (mx - BUBBLE_RADIUS, my - BUBBLE_RADIUS), (mx + BUBBLE_RADIUS, my + BUBBLE_RADIUS), (0, 0, 255), 2); cv2.line(evaluation_viz_image, (mx - BUBBLE_RADIUS, my + BUBBLE_RADIUS), (mx + BUBBLE_RADIUS, my - BUBBLE_RADIUS), (0, 0, 255), 2)
        elif marked_idx is not None and marked_coord != (-1,-1): # Single mark
            marked_x, marked_y = marked_coord
            cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS + 5, (255, 100, 0), 2) # Blue outline chosen

            # --- MODIFIED Drawing for Correct/Incorrect after removing partial correct ---
            if "Correct" in current_question_status: # Only one correct status now
                 cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (0, 255, 0), -1); cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (0, 0, 0), 1)
            # Drawing for wrong answers (remains the same)
            elif "Wrong (Full" in current_question_status: cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (0, 0, 255), -1); cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (0, 0, 0), 1)
            elif "Partially Wrong (Ring)" in current_question_status: cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (0, 0, 255), 2); cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS + RING_WIDTH, (0, 255, 255), 2)
            elif "Partially Wrong" in current_question_status: cv2.ellipse(evaluation_viz_image, (marked_x, marked_y), (BUBBLE_RADIUS, BUBBLE_RADIUS), 0, 0, 180, (0, 0, 255), -1); cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (0, 0, 0), 1)
            elif "Wrong (Low" in current_question_status: cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (128, 128, 255), 2)
            elif "Wrong (Imperfect Mark)" in current_question_status: # Visualize the imperfectly marked correct answer as wrong
                # Draw like a low confidence wrong answer? Or just the blue outline + green correct outline?
                cv2.circle(evaluation_viz_image, (marked_x, marked_y), BUBBLE_RADIUS, (128, 128, 255), 2) # Light red outline
                if correct_coord != (-1,-1): cv2.circle(evaluation_viz_image, correct_coord, BUBBLE_RADIUS + 5, (0, 255, 0), 2) # Ensure correct is shown green


        detailed_results.append((q_num, marked_option_char, correct_option_char, current_question_status, intensities))

    save_step_image(evaluation_viz_image, "08_evaluation_results")
    generate_heatmap(processed_student_image, bubble_data_for_drawing) # Saves 09 inside
    return score, evaluation_viz_image, detailed_results

# --- Main Execution Block (To run script directly) ---
if __name__ == "__main__":
    DEFAULT_IMAGE_PATH = "uploads\sample3.png" # <<<=== SET INPUT IMAGE FILE HERE
    SKIP_ALIGNMENT = False
    DEFAULT_CORRECT_ANSWERS = [1, 3, 0, 1, 1, 2] # B, D, A, B, B, C

    print(f"--- Running OMR Processing Standalone ---"); print(f"--- Target Image: {DEFAULT_IMAGE_PATH} ---")
    if not os.path.exists(DEFAULT_IMAGE_PATH): print(f"Error: Input image file not found: '{DEFAULT_IMAGE_PATH}'")
    else:
        # Step 1: Load and preprocess the image
        processed_image = load_and_preprocess_image(image_path=DEFAULT_IMAGE_PATH, skip_alignment=SKIP_ALIGNMENT)
        if processed_image is None: print("FATAL: Image preprocessing failed.")
        else:
            print(f"INFO: Preprocessing complete. Dimensions: {processed_image.shape[1]}x{processed_image.shape[0]}")
            # Step 2: Extract the marked answers
            student_answers, bubble_data = get_answers_from_image(processed_image)
            if student_answers is None: print("FATAL: Could not extract answers.")
            else:
                print(f"INFO: Extracted answers: {student_answers}")
                print(f"INFO: Using key: {DEFAULT_CORRECT_ANSWERS}")
                # Step 3: Score the sheet and generate visualizations
                score, final_image, results_list = score_student_sheet(student_answers, DEFAULT_CORRECT_ANSWERS, bubble_data, processed_image)
                # Print Final Report (Simplified print loop)
                print("\n--- OMR Evaluation Results ---"); print(f"Processed Image: {os.path.basename(DEFAULT_IMAGE_PATH)}"); print(f"Evaluated Dimensions: {processed_image.shape[1]}x{processed_image.shape[0]} (WxH)")
                print(f"Total Score: {score:.2f} / {NUM_Q * MARKS_PER_Q}"); print("-" * 80); header = f"{'Q':<3} | {'Marked':<8} | {'Correct':<8} | {'Status':<35} | {'Intensities (A, B, C, D)':<40}"
                print(header); print("-" * len(header))
                for row in results_list: print(f"{row[0]:<3} | {str(row[1]):<8} | {str(row[2]):<8} | {row[3]:<35} | {', '.join([f'{i:.3f}' for i in row[4]]):<40}")
                print("-" * len(header)); print(f"\nOutput saved in '{OUTPUT_DIR}'. Check 08_evaluation_results.png"); print(f"--- Finished: {DEFAULT_IMAGE_PATH} ---")