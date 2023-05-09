
import matplotlib.pyplot as plt

post_dates_stats = {
    "Average tweets per day": 0.024329382407985028,
    "Average tweets per week": 0.1703056768558952,
    "Average tweets per month": 0.7358490566037735,
    "Average tweets per year": 7.8,
    "Peak tweet day": "2021-04-14",
    "Peak tweet hour": "14:00",
    "Average time between tweets": "-43d -1012h 1m",
    "Minimum time between tweets": "-336d -8041h 20m 5s",
    "Maximum time between tweets": "-1d -1h 51m 12s"
}

# Plot average tweets per day, week, month, and year
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].bar(post_dates_stats.keys()[:4], post_dates_stats.values()[:4])
axs[0, 0].set_title('Average Tweets')
axs[0, 1].bar(post_dates_stats['Peak tweet day'], post_dates_stats['Average tweets per day'])
axs[0, 1].set_title('Peak Tweet Day')
axs[1, 0].bar(post_dates_stats['Peak tweet hour'], post_dates_stats['Average tweets per day'])
axs[1, 0].set_title('Peak Tweet Hour')
axs[1, 1].bar(['Avg. Time Between Tweets', 'Min. Time Between Tweets', 'Max. Time Between Tweets'], 
              [post_dates_stats['Average time between tweets'], post_dates_stats['Minimum time between tweets'], 
               post_dates_stats['Maximum time between tweets']])
axs[1, 1].set_title('Time Between Tweets')
plt.tight_layout()
plt.show()

sentiment_analysis_stats = {
    "Number of Positive Tweets": 36,
    "Number of Neutral Tweets": 1,
    "Number of Negative Tweets": 2
}

# Plot number of positive, neutral, and negative tweets
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(sentiment_analysis_stats.values(), labels=sentiment_analysis_stats.keys(), autopct='%1.1f%%')
ax.set_title('Sentiment Analysis')
plt.show()

polarity_score_stats = {
    "Average Positive Score": 0.8130833333333338,
    "Average Neutral Score": 0.0,
    "Average Negative Score": -0.6295000000000001
}

# Plot average positive, neutral, and negative scores
fig, ax = plt.subplots(figsize=(6, 6))
ax.bar(polarity_score_stats.keys(), polarity_score_stats.values())
ax.set_title('Polarity Score')
plt.show()

common_concepts = {
    "Día": 89,
    "Hoy": 70,
    "Comisión": 56,
    "Sesión": 43,
    "Inicia": 37,
    "Manera": 34,
    "Clase": 33,
    "Prof": 31,
    "Paraguay": 31,
    "Sala": 24
}

# Plot common concepts of tweets and counts
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(common_concepts.keys(), common_concepts.values())
ax.set_title('Common Concepts of Tweets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
