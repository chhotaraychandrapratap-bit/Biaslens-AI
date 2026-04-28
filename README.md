# Biaslens-AI
BiasLens AI – A multi-agent AI system for detecting, analyzing, and explaining bias in datasets. It provides fairness metrics, visual insights, and an interactive AI assistant to help users understand and mitigate bias in machine learning pipelines.
🔍 BiasLens AI

🚀 Built for Hack2Skill AI Challenge 2026

BiasLens AI is a multi-agent bias auditing system designed to detect, analyze, and explain bias in datasets used in machine learning pipelines. It helps developers and researchers identify unfair patterns and take corrective action.

---

🌟 Key Highlights

- 🔎 Detects bias across sensitive attributes (age, gender, etc.)
- 📊 Computes fairness metrics like Demographic Parity & Equal Opportunity
- 🤖 AI-powered assistant explains bias in natural language
- 📈 Interactive dashboard with visual insights
- ⚡ Lightweight, fast, and easy to use

---

🎯 Problem Statement

Machine learning models often inherit bias from training data, leading to unfair decisions. BiasLens AI helps identify and explain these biases, making AI systems more transparent and ethical.

---

🚀 Features

📊 Fairness Metrics

- Demographic Parity Difference
- Equal Opportunity Difference
- Target distribution analysis

🤖 AI Chat Assistant

- Ask questions like:
  - “Which group is most disadvantaged?”
  - “How serious is this bias?”
- Get clear explanations using LLMs

📈 Visualization

- Bar charts for distribution
- Bias indicators
- Clean UI for easy interpretation

⚙️ Multi-Agent System

- Audit Agent → Computes metrics
- Explanation Agent → Explains results
- Fix Advisor Agent → Suggests improvements

---

🛠️ Tech Stack

- Python 🐍
- Streamlit 🎨
- Pandas & NumPy 📊
- Groq API (LLM) 🤖

---

📂 Project Structure

BiasLens-AI/
│
├── app.py            # Streamlit frontend
├── metrics.py        # Bias & fairness calculations
├── agents.py         # AI agents & chat assistant
├── requirements.txt  # Dependencies
└── README.md         # Documentation

---

▶️ How to Run Locally

pip install -r requirements.txt
streamlit run app.py

---

🔐 Environment Setup

Set your Groq API key:

export GROQ_API_KEY=your_api_key_here

Or for Windows:

set GROQ_API_KEY=your_api_key_here

📊 Example Workflow

1. Upload CSV dataset
2. Select target column
3. Select sensitive attribute
4. Run audit
5. View bias metrics & charts
6. Ask AI assistant questions

---

🎯 Use Cases

- Ethical AI development
- Dataset auditing before model training
- Bias detection in hiring, finance, healthcare datasets

---

🚧 Future Improvements

- More fairness metrics (e.g., Equalized Odds)
- Model-level bias detection
- Automated bias mitigation
- Better UI/UX and dashboards

---

👨‍💻 Author

- Chandrapratap and team

---

📜 License

MIT License
