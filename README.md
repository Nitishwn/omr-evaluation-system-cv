# Simple OMR Bubble Sheet Grader

This Python script helps grade multiple-choice answer sheets (like the ones with bubbles you fill in!) using computer vision.

## What it Does

1.  **Loads an Image:** Takes a picture or scan of a completed answer sheet.
2.  **Finds the Sheet:** Tries to detect the corners of the answer sheet in the image.
3.  **Straightens It Out:** Corrects the perspective so the sheet looks flat, even if the picture was taken at an angle.
4.  **Locates the Bubbles:** Finds where all the answer bubbles are supposed to be based on predefined settings.
5.  **Checks Your Marks:** Measures how much each bubble is filled in.
6.  **Determines Answers:** Figures out which option (A, B, C, D) was marked for each question.
7.  **Scores the Sheet:** Compares the marked answers to a correct answer key and calculates the final score (including negative marking if configured).
8.  **Shows the Results:**
    * Prints a detailed report to the console.
    * Saves images showing the detected bubbles, the graded sheet with marks, and a heatmap of filled areas in an `output_steps` folder.
    * Saves a CSV file with the fill level for every single bubble.

## Tech Used

* **Python 3**
* **OpenCV (`opencv-python`)**: For all the image processing magic.
* **NumPy**: For numerical operations, especially with coordinates.
* **Pillow (PIL Fork)**: For opening different image formats.

## How to Use

1.  **Install Libraries:** If you haven't already, install the required libraries:
    ```bash
    pip install opencv-python numpy Pillow
    ```
2.  **Configure the Script:** Open the Python script (`.py` file) and find the `if __name__ == "__main__":` block near the bottom.
    * **Set Image Path:** Change `DEFAULT_IMAGE_PATH = "uploads\sample3.png"` to the actual path of the answer sheet image you want to grade.
    * **Set Answer Key:** Modify `DEFAULT_CORRECT_ANSWERS = [1, 3, 0, 1, 1, 2]` to match the correct answers for your sheet (0=A, 1=B, 2=C, 3=D).
    * *(Optional)* Adjust other constants near the top (like `NUM_Q`, `NUM_O`, thresholds, coordinates) if your sheet layout is different.
3.  **Run the Script:**
    ```bash
    python your_script_name.py
    ```
4.  **Check the Output:**
    * Look for the score and detailed results printed in your terminal.
    * Check the `output_steps` folder for the generated images (`08_evaluation_results.png` is the main graded image).

That's it! It reads the bubbles and tells you the score.
