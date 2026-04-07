# OpenEnv Hackathon Submission: Robotic Plotter Trajectory Planner

## 1. Project Overview
The environment simulates a real-world CNC machine or robotic plotter, strictly satisfying the hackathon requirement to build an environment that "must simulate a real-world task (not games or toys)" [1]. The LLM agent acts as the hardware controller, attempting to draw vector representations of letters by issuing coordinate-based stroke commands.

## 2. Tasks & Final Grading
To meet the requirement of implementing a minimum of 3 tasks with agent graders that produce scores between 0.0 and 1.0 [2]:
*   **Easy Task:** Draw straight-line letters (e.g., 'L', 'V').
*   **Medium Task:** Draw single curved letters (e.g., 'C', 'O').
*   **Hard Task:** Draw complex, multi-stroke letters with intersections (e.g., 'A', 'R').
*   **Final Automated Grader:** The final grader evaluates the overall task performance. To ensure the final score never violates the strict 0.0 to 1.0 range limit [2], the grader calculates the Intersection over Union (IoU) of the drawn path against the ground-truth 100x100 pixel grid. (Alternatively, it can use the clamped formula: `max(0.0, (correct - incorrect) / absolute_sum)`).

## 3. Step-by-Step Reward Algorithm
Inside your `step()` method, the environment returns a meaningful reward with partial progress signals [2].
*   **Reward Formula:** It evaluates `(correct guessed pixels - incorrectly guessed pixels) / absolute_sum`.
*   This provides a positive reward for successfully overlapping target pixels and utilizes **negative rewards** to actively penalize the agent for drawing on the black background.

## 4. The OpenEnv Implementation (5-Step Pattern)
*   **Step 1: Type-Safe Models (`models.py`)**: Define the `Action` (regex-parsed stroke properties), `Observation` (text feedback/reward), and `State` metadata [3].
*   **Step 2: Environment Logic (`server/environment.py`)**: Implement the standard `reset()`, `step()`, and `state()` API endpoints to handle the simulation logic [4].
*   **Steps 3 & 4: Client and Server (`client.py` & `server/app.py`)**: Create your `HTTPEnvClient` and pass the environment to `create_fastapi_app(env)` to generate HTTP endpoints [4, 5].
*   **Step 5: Dockerize (`server/Dockerfile`)**: Package the application so the automated validators can successfully build the container [5, 6].

## 5. The Inference Script (`inference.py`)
*   **Location & Hardware:** Must be named exactly `inference.py` in the root directory [7]. It must run in under 20 minutes on a machine constrained to 2 vCPUs and 8GB of memory [8].
*   **LLM Integration:** Must use the OpenAI Client [9] and utilize `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from the environment variables [7, 10].
*   **Strict Logging:** Must emit structured stdout logs strictly following the `[START]`, `[STEP]`, and `[END]` format. Any deviation results in an incorrect evaluation score [9].
*   **Regex Optimization:** Instruct the LLM via the system prompt to only output regex-parsable coordinate commands (e.g., `[PLOT: type, x1, y1, x2, y2]`) to drastically improve token speed and clear the runtime limits.

## 6. Deployment & Documentation
Before submitting your final Hugging Face Space URL [2, 11], ensure you include:
*   An `openenv.yaml` file defining the environment configuration [1].
*   A `README` detailing the environment description, action/observation spaces, and setup instructions [12].