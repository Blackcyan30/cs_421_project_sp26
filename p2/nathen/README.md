# Project Part Two: Natural Language Generation
**CS 421: Natural Language Processing**

## 1. Implementation Details

### Q1: Corpus-Based Chatbot
* **Methodology:** Implemented a retrieval-based (corpus-based) chatbot. Instead of generating new tokens, the system identifies the most appropriate response already existing in the training data using a multi-factor similarity algorithm.
* **Embeddings:** We utilized the `all-MiniLM-L6-v2` SentenceTransformer to encode both the conversation history (query) and the potential responses (corpus) into contextual vector representations.
* **Similarity Logic ($S_{total}$):** The selection process was governed by a weighted similarity function consisting of four components:
    1. **Textual Similarity ($w_1=0.4$):** Cosine similarity between the embeddings of the conversation history and the training utterances.
    2. **Emotion Intensity ($w_2=0.2$):** Mathematical similarity ($1 - |Target - Candidate|$) of normalized emotion scores.
    3. **Empathy Intensity ($w_3=0.2$):** Mathematical similarity ($1 - |Target - Candidate|$) of normalized empathy scores.
    4. **Emotional Polarity ($w_4=0.2$):** A binary check ensuring the candidate response matches the desired polarity label.
* **Generation Process:** For the test set, the model generates responses for turns 6 through 10 by evaluating the conversation history at each step against the training corpus to find the highest $S_{total}$ score.

### Q2: In-Context Learning (ICL) LLM Chatbot
* **Model:**
* **Technique:** Few-shot prompting using 3-shot and 5-shot configurations to guide the LLM's generation within the context window.
* **Task:** Generation of 10 utterances starting from turn 6 based on provided history.

---

## 2. Results on Development Set
*Evaluation conducted on turns 6-10 of the `trac2_CONVT_dev.csv` dataset.*

| Method | ROUGE-L | BLEU | BertScore (F1) |
| :--- | :--- | :--- | :--- |
| **Q1: Corpus-Based** | **0.1094** | **0.0101** | **0.8600** |
| **Q2: LLM (ICL)** |  |  |  |

---

## 3. Preprocessing and Model Choices
* **Weight Configuration:** After testing multiple configurations, we settled on $w_1=0.4, w_2=0.2, w_3=0.2, w_4=0.2$. We found that while higher weights on $w_1$ increased keyword matching, lower weights allowed the model to prioritize the emotional "vibe" requested by the task.
* **Metric Reflection:** The results show a low BLEU/ROUGE score but a high BertScore (0.86). This indicates that the corpus-based model is retrieving responses that are semantically and contextually relevant to the conversation, even if they do not contain the exact word-for-word overlap found in the ground truth.
* **Normalization:** All numerical features (Emotion/Empathy) were normalized to a 0-1 range using `MinMaxScaler` to ensure that no single feature dominated the similarity calculation.

---

## 4. Instructions to Run Code
1. **Environment:** Ensure Python 3.10+ is installed with `torch`, `sentence-transformers`, and `evaluate`.
2. **GPU Usage:** It is highly recommended to run the notebook on a **T4 GPU** in Google Colab to speed up the encoding of the training corpus.
3. **Data Path:** Place `trac2_CONVT_train.csv`, `trac2_CONVT_dev.csv`, and `project_part2_test.csv` in the same directory as the script.
4. **Execution:** Run the blocks in sequence. The first block encodes the corpus, the second defines the similarity logic, and the final blocks generate the `generations_corpus.csv` file and output the dev metrics.

---

## 5. Output Files
* `generations_corpus.csv`: Retrieval-based chatbot outputs for the test set.
* `generations_icl.csv`: LLM-generated outputs for the test set.
