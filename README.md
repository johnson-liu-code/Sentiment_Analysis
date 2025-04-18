

# Sarcasm Detector
2025 Spring\
Diablo Valley College\
Project Bracket

---

### __Authors__

Johnson Liu ( project manager )\
GitHub: [@johnson-liu-code](https://github.com/johnson-liu-code)\
Email: [liujohnson.jl@gmail.com](mailto:liujohnson.jl@gmail.com)

Heidi - [@heidi415D](https://github.com/heidi415D)

Bryan - [@hBrymiri](https://github.com/hBrymiri)

---

### __Project Overview__

#### __Introduction / Background__
... text here ...

#### __Purpose__
... text here ...

#### __Data Used__
... text here ...

#### __Variable Definitions__

See reference [2. GloVe model](#theoretical-foundations) in the Theoretical Functions section.
1. $V$ is the set of all unique words that appear in the corpus.

1. $X$ is the co-occurrence matrix for every possible pair of words $i$ and $j$ from $V$.

1. $X_{ij}$ is the $i$-th row, $j$-th column entry in $X$ which gives the number of times word $j$ appears in the context of word $i$.

1. $X_i = \sum_{k \in V} X_{ik}$ is the sum of the number of times every word $k$ appears in the context of word $i$, with the exception of word $i$. Although repeated instances of word $i$ are also counted in the context of word $i$.

1. ...
1. ....
1. more stuff ...

#### __Mathematical Foundations__
... text here ...

#### __Workflow / Pipeline__
... text here ...

#### __Results__
... text here ...

#### __Future Direction / Possible Improvements__
... text here ...
1. Extend project to sentiment and tone classificaiton of text.
1. 

---

### __Resources__

#### __Data__
1. [Kaggle dataset with Reddit posts classified as either sarcastic or not sarcastic.](https://www.kaggle.com/datasets/danofer/sarcasm/data?select=train-balanced-sarcasm.csv)

#### __Theoretical Foundations__

##### <ins>Natural language processing</ins>
1. [Natural language processing (Wikipedia article).](https://en.wikipedia.org/wiki/Natural_language_processing)

1. [Text classification and sentiment analysis (blog post).](https://mlarchive.com/natural-language-processing/text-classification-sentiment-analysis/)

1. [Word embedding (blog post).](https://towardsdatascience.com/text-embeddings-comprehensive-guide-afd97fce8fb5/)

1. [Word embedding (blog post).](https://towardsdatascience.com/word-embeddings-explained-c07c5ea44d64/)

##### <ins>word2vec model</ins>
1. [Theoretical guide for word2vec models (blog post).](https://mlarchive.com/natural-language-processing/word2vec-nlp-with-contextual-understanding/)

1. [Word2vec model (Wikipedia article).](https://en.wikipedia.org/wiki/Word2vec)

1. [Word2vec and GloVe models (blog post).](https://mlarchive.com/natural-language-processing/word2vec-nlp-with-contextual-understanding/)

1. [Continous bag of words and word2vec models (blog post).](https://medium.com/@anmoltalwar/cbow-word2vec-854a043ee8f3)

1. [*Efficient Estimation of Word Representations in Vector Space* (original academic paper).](https://arxiv.org/abs/1301.3781v3)

##### <ins>GloVe model</ins>
1. [GloVe model (Wikipedia article).](https://en.wikipedia.org/wiki/GloVe)

1. [*GloVe: Global Vectors for Word Representation* (original manusript/academic paper).](https://nlp.stanford.edu/pubs/glove.pdf)

#### __Sample Works__
1. [Project applying the word2vec and GloVe models to classifying news headlines. Models were trained using headlines from _The Onion_ and the _The Huffington Post_.](https://www.kaggle.com/code/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy)

#### __Documentation and Tutorials__

##### <ins>Neural Networks</ins>
1. [Building and training a neural network in Python with Keras (blog post on a machine learning blog).](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

1. [Another tutorial on building neural networks in Python (GeeksforGeeks).](https://www.geeksforgeeks.org/training-a-neural-network-using-keras-api-in-tensorflow/)

1. [Tutorial on building your own neural network in Python from scratch (Real Python).](https://realpython.com/python-ai-neural-network/#python-ai-starting-to-build-your-first-neural-network) 

#### Gensim
1. [Gensim word2vec tutorial (notebook posted on Kaggle by one of the developers of Gensim).](https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial)

1. [Word2vec module (documation from the Gensim website).](https://radimrehurek.com/gensim/models/word2vec.html)

1. [Gensim word2vec tutorial (documentation from the Gensim website).](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)

#### __Other Theoretical Backgrounds__
1. [General overview on machine learning (GeeksforGeeks).](https://www.geeksforgeeks.org/machine-learning/)

1. [General overview on artificial intelligence, machine learning, and data science (GeeksforGeeks).](https://www.geeksforgeeks.org/ai-ml-ds/)

1. [Bag of words model (Wikipedia article).](https://en.wikipedia.org/wiki/Bag-of-words_model)

1. [Logistic regression (Wikipedia article).](https://en.wikipedia.org/wiki/Logistic_regression)

1. [Multinomial logistic regression (Wikipedia article).](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

1. [Least squares (Wikipedia article).](https://en.wikipedia.org/wiki/Least_squares)

1. [Tf-idf [ term frequency-inverse document frequency ] (Wikipedia article).](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

#### __Mathematical References__

1. [Dot product (Wikipedia article).](https://en.wikipedia.org/wiki/Dot_product)

1. [Cosine similarity (Wikipedia article).](https://en.wikipedia.org/wiki/Cosine_similarity)

1. [Linear least squares (Wikipedia article).](https://en.wikipedia.org/wiki/Linear_least_squares)

---

### __Important Dates ( taken from club-provided syllabus )__

##### <ins>Week 5/6 — April 16 & April 23, 2025</ins>
Development continues on in week 5 in preparation for the **mid-semester showcase in week 6**. Groups are now in the middle of the semester meaning that they will present what progress they have so far. The mid-semester showcase does not mean that groups have to be halfway done with their projects.

##### <ins>Week 7 — April 30, 2025</ins>
At this point **groups should be more than halfway done with their project** or close to finished in preparation for the final week as well as finals. Project managers should check with members on the final schedule to ensure projects are done and not rushed in the final week.

##### <ins>Week 8 — May 7, 2025</ins>
**Groups should be close to wrapping up their projects** or should be completely done with the projects. This week will be focused mainly on the final project showcase in which judges will determine who has the best project. Groups may want to **prepare ahead of time with presentations**, graphics, and props to enhance their presentations.

---
### __Preliminary Data Visualization__

#### Word cloud - Sarcastic
![placeholder-text](data_visualization/wordcloud_sarcastic.png)

#### Word cloud - Not Sarcastic
![placeholder-text](data_visualization/wordcloud_not_sarcastic.png)

#### Word frequency in comments
![placeholder-text](data_visualization/words_in_comments.png)

---

### __Records__

---

#### <ins>20250404</ins>

##### Notes
— Johnson
1. Heidi and Brymiri, I will be using this file to keep track of out progress and to assign tasks amongst the team members.
2. I will also use this file to keep track of important dates specified by the club.
3. Please use this file to record any important thoughts that you come up with during the course of the project.
4. You can also use this file to add important notes that you want the team to remember.

##### To-do —
###### For Johnson:
- [x] Create Github repository.
- [x] Look up relavant resources.
- [x] Look up relavant data.
- [ ] Plan out our timeline and general to-do's / tasks for each team member.

---

#### <ins>20250408</ins>

##### To-do —
###### For Heidi:
- [x] Review the theoretical resources and begin drafting a short write-up for our team to reference.
- [ ] Optional - Also look up introductory machine learning resources to clarify specific topics in the write-up, especially if you think they’d be helpful for the team.

###### For Bryan:
- [ ] Write code to extract a specific Reddit post, along with its parent post and the name of the subreddit where it was posted.

---

#### <ins>20250412</ins>

##### Notes
— Johnson
1. Organized files into folders based on their purpose for better structure and clarity.
2. Expanded and refined the files used for data extraction.
3. Developed files dedicated to data visualization.
4. Generated figures to help us better understand the data.

---

#### <ins>20250413</ins>

##### Notes
— Johnson
1. Enhanced `extract_data.py` with docstring and example usage.
1. Updated comments in `get_relevant_data.py`.
1. Added `cooccurrence_matrix_heatmap.png` for visualization.
1. Refactored `generate_wordcloud.py` and `visualize_comment_length.py` to include main guard and improved comments.
1. Introduced `cooccurrence_matrix.py` for creating and plotting co-occurrence matrices.
1. Added `cooccurrence_probability.py` for calculating co-occurrence probabilities.
1. Implemented `word_vectors.py` for generating random word vectors.
1. Added `test.py` for testing co-occurrence matrix functionality.
1. Removed obsolete `word2vec.py`.

---

#### <ins>20250418</ins>

##### To-do —
###### For Heidi:
- [ ] Experiment / explore with creating and training neural networks.
- [ ] Familiarize yourself with the use of the Gensim Python module for word vectors.
- [ ] Start building and training neural networks to take in word vectors as input and outputs a binary classification ( sarcastic or not sarcastic ).

###### For Bryan:
- [ ] Continue working on code to extract a specific comment from Reddit along with its parent comment and subreddit name.
- [ ] Possible future feature: create an app that can be used in the browser to predict whether a selected comment is sarcastic or not.

---

\
\
\
\
\
\
\
\
\

---
### __. . . Test / Miscellaneous  . . .__

#### Markdown supports LaTeX formatting! :D

**The Cauchy-Schwarz Inequality**

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
