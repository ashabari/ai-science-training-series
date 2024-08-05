import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import sys
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['figure.figsize'] = [12, 8]  # Adjusted figure size

# Load BERT.
model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking')
model.eval()  # Set the model to eval mode

# Load tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

# Save the BERT vocabulary to a file.
tokenizer.save_vocabulary(save_directory='.')

print("The vocabulary size is: ", model.config.vocab_size)  # Size of the vocabulary

# Get BERT's vocabulary embeddings.
wordembs = model.get_input_embeddings()

# Convert the vocabulary embeddings to numpy.
allinds = np.arange(0, model.config.vocab_size, 1)
inputinds = torch.LongTensor(allinds)
bertwordembs = wordembs(inputinds).detach().numpy()
print(bertwordembs.shape)

# Read in the vocabulary
filename = "vocab.txt"
with open(filename, 'r') as f:
    bertwords = np.array([line.rstrip() for line in f])

# Determine vocabulary to use for t-SNE/visualization
bert_char_indices_to_use = np.arange(999, 1063, 1)
bert_voc_indices_to_plot = np.append(bert_char_indices_to_use, np.arange(1996, 5932, 1))
bert_voc_indices_to_use = np.append(bert_char_indices_to_use, np.arange(1996, 11932, 1))

bert_voc_indices_to_use_tensor = torch.LongTensor(bert_voc_indices_to_use)
bert_word_embs_to_use = wordembs(bert_voc_indices_to_use_tensor).detach().numpy()
bert_words_to_plot = bertwords[bert_voc_indices_to_plot]

print(len(bert_voc_indices_to_plot))
print(len(bert_voc_indices_to_use))

# Run t-SNE on the BERT vocabulary embeddings we selected
mytsne_words = TSNE(n_components=2, early_exaggeration=12, verbose=2, metric='cosine', init='pca', n_iter=2500)
bert_word_embs_to_use_tsne = mytsne_words.fit_transform(bert_word_embs_to_use)

# Plot the transformed BERT vocabulary embeddings
fig, ax = plt.subplots()
alltexts = list()
for i, txt in enumerate(bert_words_to_plot):
    ax.scatter(bert_word_embs_to_use_tsne[i, 0], bert_word_embs_to_use_tsne[i, 1], s=0)
    currtext = ax.text(bert_word_embs_to_use_tsne[i, 0], bert_word_embs_to_use_tsne[i, 1], txt, family='sans-serif')
    alltexts.append(currtext)

# Save the plot before adjusting
plt.savefig('visualizingTokenEmbedding.pdf', format='pdf')

# Display the plot
plt.show()

nplotted = len(bert_words_to_plot)

# Get the limits of the embedding region
x_left = np.min(bert_word_embs_to_use_tsne[:nplotted,0])
x_right = np.max(bert_word_embs_to_use_tsne[:nplotted,0])
y_left = np.min(bert_word_embs_to_use_tsne[:nplotted,1])
y_right = np.max(bert_word_embs_to_use_tsne[:nplotted,1])

# Width of embedding region
xwidth = x_right - x_left ; ywidth = y_right - y_left


def plotwindow(window_xfrac, window_yfrac, window_xoff, window_yoff):
    """
    Plot a requested window on the embedding.

    window_xfrac (float): Fraction of the X-window size to plot
    window_yfrac (float): Fraction of the Y-window size to plot
    window_xoff (float): X-offset of window center, from the left. Should be in [0.0, 1.0],
        and is interpreted as a fraction of the window X-width
    window_yoff (float): Y-offset of window center, from the bottom. Should be in [0.0, 1.0],
        and is interpreted as a fraction of the window Y-width
    
    """

    ## Bounds on window coordinates
    newxl_center = x_left + xwidth * window_xoff
    newxl_left = newxl_center - 0.5*window_xfrac * xwidth
    newxl_right =  newxl_center + 0.5*window_xfrac * xwidth

    newyl_center = y_left + ywidth * window_yoff
    newyl_left = newyl_center - 0.5*window_yfrac * ywidth
    newyl_right =  newyl_center + 0.5*window_yfrac * ywidth

    # Make arrays including only words inside the requested window, using numpy
    # logical indexing
    nplotted = len(bert_words_to_plot)
    x_inds_r = bert_word_embs_to_use_tsne[:nplotted,0] < newxl_right 
    x_inds_l = bert_word_embs_to_use_tsne[:nplotted,0] > newxl_left
    x_inds = np.logical_and(x_inds_l, x_inds_r)

    y_inds_r = bert_word_embs_to_use_tsne[:nplotted,1] < newyl_right
    y_inds_l = bert_word_embs_to_use_tsne[:nplotted,1] > newyl_left
    y_inds = np.logical_and(y_inds_l, y_inds_r)

    inds = np.logical_and(x_inds, y_inds)

    # Index the word and embedding arrays by the window-restricted indices
    bwtp = bert_words_to_plot[inds]
    bwembs = bert_word_embs_to_use_tsne[:nplotted,:][inds,:]
    
    # Plot away
    for i, txt in enumerate(bwtp):
        plt.scatter(bwembs[i,0], bwembs[i,1], s=0)
        plt.text(bwembs[i,0], bwembs[i,1], txt, family='sans-serif', size=75)

    plt.show()

window_xfrac = 0.1 ; window_yfrac = 0.1
window_xoff = 0.14 ; window_yoff = 0.45  ## Place names
print("Place Names")
plotwindow(window_xfrac, window_yfrac, window_xoff, window_yoff)

window_xfrac = 0.05 ; window_yfrac = 0.05
window_xoff = 0.92 ; window_yoff = 0.42  ## Time relations
print("Time relations")
plotwindow(window_xfrac, window_yfrac, window_xoff, window_yoff)

