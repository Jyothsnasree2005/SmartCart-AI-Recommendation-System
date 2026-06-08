# SmartCart AI - Personalized E-Commerce Recommendation System

## Project Overview

SmartCart AI is an end-to-end recommendation system developed using Collaborative Filtering with Alternating Least Squares (ALS). The system provides personalized product recommendations for customers based on purchase history.

## Features

* Personalized Product Recommendations
* Trending Product Analysis
* PostgreSQL Database Integration
* FastAPI REST APIs
* Streamlit Interactive Dashboard
* ALS Collaborative Filtering Model

## Technology Stack

* Python
* Pandas
* Scikit-Learn
* FastAPI
* PostgreSQL
* Streamlit
* SciPy

## System Architecture

Dataset → PostgreSQL → ALS Model → FastAPI → Streamlit Dashboard

## API Endpoints

### Home

GET /

### Trending Products

GET /trending

### Personalized Recommendations

GET /recommend/{customer_id}

## Dataset Statistics

* Customers: 1.37 Million
* Products: 105 Thousand
* Transactions: 500 Thousand

## Results

The recommendation engine successfully generates personalized product suggestions for customers based on historical purchase behavior.

## Future Enhancements

* Item2Vec Embeddings
* FAISS Similarity Search
* Redis Caching
* Hybrid Recommendation Engine

## Author

Gunda Jyothsna Sree

