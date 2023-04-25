# Mid Bootcamp Project: Movie Analysis

This project is a movie analysis based on data from IMDb.com, incorporating a large sample of movies I have seen. The goal is to explore data science tools learned in the bootcamp and have fun with the data. The ultimate goal is to create a recommendation system for movies based on my taste.

## Table of Contents
1. [Project Description](#project-description)
2. [Data Collection](#data-collection)
3. [Exploratory Data Analysis (EDA)](#eda)
4. [Data Preprocessing & Feature Engineering](#data-preprocessing)
5. [Data Transformation](#data-transformation)
6. [Visualizations](#visualizations)
7. [Hypothesis Testing](#hypothesis-testing)
8. [Custom Overview: My Watched Movies](#custom-overview)
9. [Recommendation System](#recommendation-system)

## Project Description

I've kept track of the movies I watch since 2015 on Pinterest: [My Watched Movies Board](https://www.pinterest.com/chaosskill/movies-watched/). To explore the data, I have collected data on movies from IMDb, Pinterest, Kaggle, and Apify, among other sources.

I will investigate a few questions, such as:

- What are the most popular genres and how do they change over time?
- Which actors and directors are the most successful?
- Is there a correlation between award-winning actors and movie ratings?

Ultimately, I aim to create a movie recommendation system based on my personal taste.

## Data Collection

- Pinterest Board: Scraped using Apify's Pinterest crawler
- IMDb dataset: Found a large database on Kaggle
- Oscars and Golden Globe winners: Found datasets on Kaggle

## Exploratory Data Analysis (EDA)

In this section, I dive deeper into the exploration of the movie data.

## Data Preprocessing & Feature Engineering

I analyze IMDb ratings, introduce weighted ratings, and account for the time decay factor. I also cast directors, genre and starts to lists so I can work with them more easily.

## Data Transformation

I perform One Hot Encoding for the 'genre' column to prepare the data for analysis.

## Visualizations

A variety of visualizations are used to better understand the movie data, such as:

- Runtime distribution
- Number of movies per genre
- Average weighted rank by genre
- Top actors by frequency in the top 5000 movies
- Top directors by frequency in the top 5000 movies

## Hypothesis Testing

I perform hypothesis testing on the correlation between award-winning actors and movie performance.

## Custom Overview: My Watched Movies

In this section, I analyze my personal movie-watching habits, focusing on:

- Top actors in the movies I've watched
- Comparison between my taste and average ratings
- Insights from my Pinterest board

## Recommendation System

To create a movie recommendation system based on my taste, I implement:

- Content-based filtering using natural language processing (NLP)
- Tokenization and vectorization to work with movie descriptions
- Cosine similarity to identify similar movies
- Principal component analysis (PCA) to visualize similarity clusters in 2D and 3D

Through this project, I explore various aspects of the movie industry and my personal taste, ultimately creating a recommendation system tailored to my preferences.