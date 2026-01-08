# Project title: FinanceInsight
Named Entity Recognition (NER) for Automated Financial Data Extraction

# Executive Summary
FinanceInsight is an AI-powered financial document intelligence system that automates the extraction of named entities and financial metrics from unstructured documents. 
The system combines domain-specific Named Entity Recognition (NER) using FinBERT, contextual understanding through Google's Gemini 2.5 Flash LLM.

# Problem Solved:
Manual extraction of financial data from documents is time-consuming, error-prone, and does not scale. 
Financial analysts spend significant time locating key metrics like revenue, earnings, market capitalization, and financial ratios from lengthy reports.

# Solution Approach : 3-Layer Hybrid Architecture
Layer 1: FinBERT NER for entity detection (company names, monetary values, dates, percentages)
Layer 2: Gemini LLM for context understanding and relationship extraction
Layer 3: Gemini LLM for human-readable formatting and validation
Validation Layer : Evaluates extraction quality by comparing results with expected financial document structure. Produces a completeness score (0 - 100%) and lists missing or incomplete items.



