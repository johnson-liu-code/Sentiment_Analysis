

# Sarcasm Detector
2025 Spring\
Diablo Valley College\
Project Bracket

---

### __Authors__

Heidi\
[@heidi415D](https://github.com/heidi415D)

Brymiri\
[@hBrymiri](https://github.com/hBrymiri)

Johnson Liu ( project manager ) \
[@johnson-liu-code](https://github.com/johnson-liu-code)

---

### __Project Overview__

#### __Introduction__
\
\
\
\
\
\
\
\
\

#### __Variable Definitions__

See reference [2. GloVe model](#theoretical-foundations) in the Theoretical Functions section.
1. $V$ is the set of all unique words that appear in the corpus.
1. $X$ is the co-occurrence matrix for every possible pair of words $i$ and $j$ from $V$.
1. $X_{ij}$ is the $i$-th row, $j$-th column entry in $X$ which gives the number of times word $j$ appears in the context of word $i$.
1. $X_i = \sum_{k \in V} X_{ik}$ is the sum of the number of times every word $k$ appears in the context of word $i$, with the exception of word $i$. Although repeated instances of word $i$ are also counted in the context of word $i$. 
1. ...
1. ....
1. more stuff ...

---

### __Resources__

#### __Data__
1. [Kaggle dataset with Reddit posts classified as either sarcastic or not sarcastic.](https://www.kaggle.com/datasets/danofer/sarcasm/data?select=train-balanced-sarcasm.csv)

#### __Theoretical Foundations__
##### <ins>Natural language processing</ins> –
1. [Wikipedia article on Natural language processing.](https://en.wikipedia.org/wiki/Natural_language_processing)
2. [Blog post about text classification and sentiment analysis on a machine learning website.](https://mlarchive.com/natural-language-processing/text-classification-sentiment-analysis/)

##### <ins>word2vec model</ins> –
1. [Wikipedia article on the word2vec model.](https://en.wikipedia.org/wiki/Word2vec)
2. [Blog post about the word2vec and GloVe models on a machine learning website.](https://mlarchive.com/natural-language-processing/word2vec-nlp-with-contextual-understanding/)

##### <ins>GloVe model</ins> –
1. [Wikipedia article on the GloVe model.](https://en.wikipedia.org/wiki/GloVe)
2. [Manusript/paper from Stanford - *GloVe: Global Vectors for Word Representation* ( Pennington, Socher, Manning; 2014. ).](https://nlp.stanford.edu/pubs/glove.pdf)

#### __Sample Works__
1. [Project applying the word2vec and GloVe models to classifying news headlines. Models were trained using headlines from _The Onion_ and the _The Huffington Post_.](https://www.kaggle.com/code/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy)

#### __Other Theoretical Backgrounds__

1. [Wikipedia article on the Bag of words model.](https://en.wikipedia.org/wiki/Bag-of-words_model)
2. [ Wikipedia article on Logistic regression.](https://en.wikipedia.org/wiki/Logistic_regression)
3. [Wikipedia article on Multinomial logistic regression.](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

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
- [ ] Review the theoretical resources and begin drafting a short write-up for our team to reference.
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
## __. . . Test\Miscellaneous  . . .__

#### Markdown supports LaTeX formatting! :D

**The Cauchy-Schwarz Inequality**

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
