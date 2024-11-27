# PubMed RCT Classifier

This project is a machine learning model designed to classify **randomized controlled trials (RCTs)** from PubMed abstracts. The model processes text data from PubMed and categorizes the RCTs into different groups based on predefined labels. This project is intended for researchers and practitioners working with biomedical literature, particularly those involved in clinical trials.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to implement a classifier for identifying randomized controlled trials (RCTs) in PubMed. Using a combination of Natural Language Processing (NLP) techniques and machine learning models, the system extracts relevant information from PubMed abstracts and categorizes them into different RCT-related classes.

### Key Features:
- Text preprocessing pipeline
- Data cleaning and tokenization
- Implementation of machine learning models for text classification
- Model evaluation metrics for classification performance

## Requirements
- Python 3.7+
- Libraries: 
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow` 
  - `matplotlib` (for visualizations)
  - `seaborn`
  - `requests` (for data fetching)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pubmed-rct-classifier.git
   cd pubmed-rct-classifier
