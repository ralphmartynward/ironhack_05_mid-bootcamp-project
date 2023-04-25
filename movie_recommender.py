import streamlit as st
import plotly.express as px
import pandas as pd

# Your data loading and processing logic here, e.g.:
movies_clean = pd.read_csv("data/clean/movies_clean.csv")

# What are the directors that have the highest average weighted rank?
import numpy as np
import matplotlib.pyplot as plt

N = 100  # Number of top directors to display

movies_clean['director_list'] = movies_clean['director'].apply(lambda x: x.split(', '))

exploded_moviess = movies_clean.explode('director_list')

director_stats_all = exploded_moviess.groupby('director_list').agg({
    'votes': ['count', 'mean']
}).reset_index()

# Rename the columns
director_stats_all.columns = ['director', 'frequency', 'votes']

# Sort the actors by their average adjusted_votes in descending order
director_stats_sorted = director_stats_all.sort_values('votes', ascending=False)

# Select the top N actors
top_directors = director_stats_sorted.head(N)
# Remove rows with 'avg_weighted_rank' equal to zero
top_directors = top_directors[top_directors['votes'] != 0]
top_directors = top_directors[top_directors['frequency'] >= 4]

# Now create the treemap
import plotly.express as px

fig = px.treemap(
    top_directors,
    path=['director'],
    values='votes',
    color='frequency',
    color_continuous_scale='Greens',
    title=f"Top Directors by average top rankings",
    width=1500,
    height=800,
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title="Frequency",
    )
)

# Display the chart in Streamlit
st.plotly_chart(fig)
