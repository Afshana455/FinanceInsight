# Project title: FinanceInsight
Named Entity Recognition (NER) for Automated Financial Data Extraction

# Executive Summary
FinanceInsight is an AI-powered financial document intelligence system that automates the extraction of named entities and financial metrics from unstructured documents. <br>
The system combines domain-specific Named Entity Recognition (NER) using FinBERT, contextual understanding through Google's Gemini 2.5 Flash LLM.

# Problem Solved: 
Manual extraction of financial data from documents is time-consuming, error-prone, and does not scale. <br>
Financial analysts spend significant time locating key metrics like revenue, earnings, market capitalization, and financial ratios from lengthy reports.

# Solution Approach : 3-Layer Hybrid Architecture
Layer 1: FinBERT NER for entity detection (company names, monetary values, dates, percentages)

Layer 2: Gemini LLM for context understanding and relationship extraction

Layer 3: Gemini LLM for human-readable formatting and validation

Validation Layer : Evaluates extraction quality by comparing results with expected financial document structure. Produces a completeness score (0 - 100%) and lists missing or incomplete items.


# Architecture Overview
The system follows a layered architecture:
1.	Presentation Layer: Streamlit web interface for file upload and results display
2.	Document Processing Layer: Format-specific text extraction (PyPDF2, python-docx, EasyOCR)
3.	Core Extraction Layer: 3-layer NER pipeline (FinBERT + Gemini LLM)
4.	Output Layer: Structured entity display with metrics and download option


<img width="934" height="452" alt="Screenshot 2026-01-08 232256" src="https://github.com/user-attachments/assets/8427ea0f-c6c0-42fa-98ae-3756b88130e9" />

# Technical Stack

| Component           | Technology              | Version          | Purpose                   |
|--------------------|-------------------------|------------------|---------------------------|
| UI Framework       | Streamlit               | 1.50.0           | Web interface             |
| NER Model          | FinBERT                 | Fine-tuned       | Entity detection          |
| LLM                | Gemini 2.5 Flash        | API              | Context & formatting      |
| Deep Learning      | Torch                   | 2.5.1+cu121      | GPU inference             |
| OCR                | EasyOCR                 | 1.7.2            | Image text extraction     |
| OCR Fallback       | pytesseract             | 0.3.13           | Backup OCR                |
| Document Parsing   | PyPDF2, python-docx     | 3.0.1, 0.2.4     | PDF / Word extraction     |

# Target Users
•	Financial Analysts: Extract key metrics from quarterly/annual reports <br>
•	Investment Researchers: Compare data across multiple companies <br>
•	Auditors & Accountants: Verify financial statement data <br>
•	Students & Researchers: Learn financial analysis through structured extraction <br>




