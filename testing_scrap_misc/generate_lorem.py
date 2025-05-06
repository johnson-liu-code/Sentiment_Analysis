

import lorem

s = lorem.sentence()

sentences = [lorem.sentence() for _ in range(5000)]

# Save sentences to a text file.
with open('sentences.txt', 'w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')