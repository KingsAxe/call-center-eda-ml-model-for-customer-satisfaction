**Project Overview**

This repository contains an end-to-end machine learning workflow for analyzing customer interaction data and classifying customer intents from speech transcripts. The project demonstrates how exploratory data analysis, unsupervised clustering, NLP-based intent modeling, and UI deployment can be combined into a practical analytics solution.

All data used here is synthetic and anonymized. The workflow mirrors a real-world production pipeline but does not include any client-sensitive information.

**Objectives**

- Understand customer interaction behavior through exploratory data analysis
- Identify operational patterns using unsupervised clustering

- Build an NLP model to classify customer intent from conversation text

- Deploy a lightweight Streamlit app for real-time inference and testing

**Key Insights**

- Phone support produces the majority of follow-ups, suggesting higher complexity or less consistent resolution compared to digital channels

- Clustering reveals recurring interaction archetypes such as routine resolution, abandoned calls, escalations, and high-quality resolutions

- NLP modeling converts speech transcripts into structured business intents for monitoring and triage

**Technical Stack**

**Core**

Python

pandas, numpy

scikit-learn

**Clustering**

Gaussian Mixture Models

HDBSCAN

**NLP**

TF-IDF Vectorization

Linear classifier

Deployment

**Streamlit**

Running the Application
Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows


**Install dependencies:**
pip install -r requirements.txt

**Run Streamlit:**

streamlit run src/app.py

**Disclaimer**

This repository is intended for educational and portfolio demonstration purposes.
The dataset is an ethical replical from a previous client project, and no confidential business data is included.

**Contribution**

Contributions, discussion, and iterations are welcome.
