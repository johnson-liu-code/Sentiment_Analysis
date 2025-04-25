

import lorem

s = lorem.sentence()
# print(s)

sentences = [lorem.sentence() for _ in range(5000)]
# print(sentences)

# Save sentences to a text file
with open('sentences.txt', 'w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')