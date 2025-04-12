

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Define the output CSV file name
data_file_name = '../data/trimmed_training_data.csv'

# Read the CSV file into a DataFrame.
df = pd.read_csv(data_file_name)

# Ensure all elements in the 'comment' and 'parent_comment' columns are strings.
df['comment'] = df['comment'].astype(str)
df['parent_comment'] = df['parent_comment'].astype(str)

# Make a word cloud for the non-sarcastic comments.
plt.figure( figsize = (20, 20) )
wc = WordCloud( max_words = 2000, width = 1600, height = 800 ).generate( " ".join( df[df.label == 0].comment ) )
plt.imshow( wc, interpolation = 'bilinear' )

# Save the figure to a file.
plt.savefig( 'wordcloud_not_sarcastic.png', dpi = 300, bbox_inches = 'tight' )

# Make a word cloud for the sarcastic comments.
plt.figure( figsize = (20, 20) )
wc = WordCloud( max_words = 2000, width = 1600, height = 800 ).generate( " ".join( df[df.label == 1].comment ) )
plt.imshow( wc, interpolation = 'bilinear' )

# Save the figure to a file.
plt.savefig( 'wordcloud_sarcastic.png', dpi = 300, bbox_inches = 'tight' )