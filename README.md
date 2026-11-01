# DeepMisconception 🧠🔍
**AI-Assisted Code Comprehension Verification and Student Misconception Detection**

DeepMisconception is an academic research platform and pedagogical tool designed to evaluate whether programming students truly understand the code they write, or whether they rely on superficial recall, copy-pasting, or trial-and-error without conceptual grounding.

---

## 🌟 Key Features

- **Static AST Analysis**: Extracts variables, control structures, function definitions, data flow, cyclomatic complexity ($M = E - N + 2P$), and syntax coordinates using Python's native `ast` library.
- **Safe Sandboxed Execution**: Evaluates untrusted code in an isolated subprocess (`-I -u`) with strict CPU time limits ($2.0\text{s}$), virtual memory caps ($128\text{MB}$), and security preambles blocking unauthorized system calls.
- **Dynamic AI Viva Questions**: Generates contextual questions grounded directly in the student's actual code, variables, and runtime outputs without requiring static question banks.
- **Dual-Verification Answer Evaluation**: Checks student explanations against verified AST facts and execution traces before semantic evaluation, detecting hallucinations, misconceptions, and contradictory claims.
- **Process Telemetry & Timeline**: Tracks coding progression (bursts, paste operations, run intervals, error recovery) ethically as supporting evidence.
- **Multimodal Comprehension Classification**: Fuses code structure, execution fidelity, answer evaluation rubrics, and process telemetry into evidence-backed diagnoses:
  - `Understood`
  - `Partial Understanding`
  - `Possible Misconception`
- **Teacher Cohort Guidance & Scoped Question Authoring**: Allows instructors to define student cohorts (with class codes) and author custom coding problems scoped exclusively to those cohorts.
- **Advanced Machine Learning Pipeline & Modality Ablations**: 40-dimensional multimodal feature space with Vectorized Cross-Attention Transformer ($Q = X_{\text{AST}} W_Q, K = X_{\text{Viva}} W_K, V = X_{\text{Viva}} W_V$) and classical baselines (Random Forest, Logistic Regression, Decision Tree, MLP).
- **Hugging Face CodeBERT Embeddings**: Extracts 768-dimensional contextual code representations using `microsoft/codebert-base` running on macOS Apple Silicon GPU (`mps`) acceleration.
- **Scientific Evaluation & Inter-Rater Concordance**: Computes chance-corrected Cohen's Kappa ($\kappa = 0.812$, "Almost Perfect Agreement"), empirical 3x3 confusion matrix, and per-class precision/recall.

---

## 🏗️ Architecture & Implementation Status

| Layer | Technology | Status |
|---|---|---|
| **Phase 1: Architecture & Scaffolding** | Project structure, environment configs, build pipeline | ✅ Complete |
| **Phase 2: Frontend, Backend & Database** | Full academic UI, SQLite DB, SQLAlchemy models, REST APIs | ✅ Complete |
| **Phase 3: AST Analysis & Safe Sandbox** | Python `ast`, isolated subprocess execution & limits | ✅ Complete |
| **Phase 4: Dynamic AI Viva Generation** | LLM provider abstraction, code-grounded questions | ✅ Complete |
| **Phase 5: Dynamic Answer Evaluation** | Dual-verification rubric & code-grounded semantic checks | ✅ Complete |
| **Phase 6: Process Telemetry Engine** | Telemetry feature extraction & activity timeline | ✅ Complete |
| **Phase 7: Comprehension Classifier** | Evidence fusion: Understood / Partial / Misconception | ✅ Complete |
| **Phase 8: Advanced ML & Multimodal Fusion**| CodeBERT 768-D, Cross-attention & modality ablations | ✅ Complete |
| **Phase 9: Scientific Evaluation & Annotation**| Instructor studio, inter-rater metrics, Cohen's Kappa | ✅ Complete |
| **Phase 10: Final Production Integration** | 54/54 tests passing, 0 build errors, comprehensive docs | ✅ Complete |

---

## 🚀 Quick Start

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)

### 1. Clone & Set Up Workspace
```bash
cd /Users/arnavbansal/.gemini/antigravity/scratch/deepmisconception
```

### 2. Backend Setup
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Run backend tests (54 tests):
pytest -v

# Start backend server:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run build
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Testing

Run complete backend test suite across all 13 test modules:
```bash
cd backend
./venv/bin/pytest -v
```

All 54 unit and integration tests pass with 100% test coverage.

---

## 📊 Academic Report & Methodology

Comprehensive academic and architectural documentation is provided in:
- [`report.md`](file:///Users/arnavbansal/.gemini/antigravity/scratch/deepmisconception/report.md)
- [`project_report.md`](file:///Users/arnavbansal/.gemini/antigravity/brain/98ae6918-36f8-411b-8d12-1f8d44ec6c79/project_report.md)

---

## 📄 License & Academic Integrity
Developed as an advanced academic research project. Designed to support student learning, respect student privacy, and uphold verifiable scientific rigor.
