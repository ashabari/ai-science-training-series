import string
import re
from operator import itemgetter

# Create a translation table to remove punctuation
translator = str.maketrans('', '', string.punctuation)

# Initialize word dictionary
wdict = {}

# Process the text file
with open('Life_On_The_Mississippi.txt', 'r') as L:
    line = L.readline()
    while line:
        # Remove non-ASCII characters
        line = re.sub(r'[^\x00-\x7F]', '', line)
        # Remove punctuation and convert to lowercase
        line = line.translate(translator).lower()
        
        # Split the line into words and count occurrences
        words = line.split()
        for word in words:
            if word:  # Check if the word is not an empty string
                if wdict.get(word) is not None:
                    wdict[word] += 1
                else:
                    wdict[word] = 1
        
        line = L.readline()

# Part 2: Sort the dictionary by word frequency
sorted_wdict = sorted(wdict.items(), key=itemgetter(1), reverse=True)

# Vocabulary size and token size
vocabulary_size = len(wdict)
total_tokens = sum(count for word, count in sorted_wdict)
token_size = len(sorted_wdict)

# Calculate coverage for top 90%
total_word_count = sum(count for word, count in sorted_wdict)
cumulative_count = 0
distinct_words = 0

for word, count in sorted_wdict:
    cumulative_count += count
    distinct_words += 1
    if cumulative_count / total_word_count >= 0.90:
        break

# Print vocabulary size, total number of tokens, and percentage of vocabulary size
print(f"Vocabulary size = {vocabulary_size}")
print(f"Total number of tokens = {total_tokens}")
print(f"Number of tokens comprising 90% of the corpus = {distinct_words}")
percentage_of_vocabulary = (distinct_words / vocabulary_size)
print(f"This is ({distinct_words} / {vocabulary_size}) = {percentage_of_vocabulary:.6f} of the vocabulary size.")

# Print the top 100 most frequent words
print("\nTop 100 most frequent words:")
nitem = 0
maxitems = 100
for item in sorted_wdict:
    nitem += 1
    print(item)
    if nitem == maxitems:
        break

# Print the number of distinct words making up the top 90% of occurrences
print(f"\nNumber of distinct words making up the top 90% of occurrences: {distinct_words}")

