###### Natural Language Processing

Natural Language Processing (NLP) can be used for sentiment analysis, which helps identify whether a piece of text expresses a positive, negative, or neutral emotion. NLP allows us to organize unstructured data—like social media posts, emails, and other forms of natural text—into a form that machines can work with by applying text classification techniques.

This is usually done by training a classifier on a dataset that has been labeled in advance. For example, some texts might be labeled as sarcastic while others are not. The classifier learns patterns in the language based on these labels and tries to apply what it learns to new, unseen data.

###### Word Embedding
ML algorthims struggle to work well with words. Instead we will work with numbers by assigning random numbers to words. Since words can be used in different contexts, like in *sarcasm* assigning certain words more than one number is best practice so that the **Neural Network** can adjust based on context.

###### Word2Vec 
Word2Vec is a model that learns word meanings based on the words that commonly appear around them. For example, the words “*doctor*” and “*nurse*” often show up in similar contexts, so Word2Vec places them close together in vector space. This helps the model understand how words relate to one another. In sarcasm detection, understanding context is key, and Word2Vec helps capture that by noticing patterns in how words are used together.

###### GloVe
 Global Vectors for Word Representation, is another word embedding model. Unlike Word2Vec, which focuses on small windows of text, GloVe looks at word co-occurrence across an entire dataset. This gives it a more global understanding of word relationships. GloVe can help highlight unusual or ironic word pairings—something that often happens in sarcastic sentences.

Some users on Kaggle have successfully used Word2Vec and GloVe to detect sarcasm in news headlines. In one project, they trained their model using headlines from The Onion (a satirical site) and The Huffington Post (a more traditional news source). The model achieved around 83% accuracy in detecting sarcasm.

We can apply a similar approach in our project using Reddit posts. Instead of news headlines, we’ll train our model on a dataset of labeled Reddit comments, using Word2Vec or GloVe to create embeddings. This will allow our classifier to learn the difference between sincere and sarcastic language based on context—just like in the Kaggle example.

