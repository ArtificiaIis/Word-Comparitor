# You can first read the file and understand its structure
with open('./glove.6B.200d.txt') as f:
    lines = f.readlines()
for i, line in enumerate(lines[:5]):
    print('--- Line', i)
    print(line)
print("Total length:", len(lines))

# Coding Task 1: complete the `load_word_emb_dict` function to load the word embeddings into a dictionary.
# The return value of the function should be a dictionary of 400k entries. Each entry should have a key of a word, and a value of a 200-dimensional torch tensor.
# Note: remember to import the necessary packages like torch.
def load_word_emb_dict():
    # TODO

word_emb_dict = load_word_emb_dict()

# Run this code block to make sure your code is working as expected
def check_load_word_emb_dict():
    assert len(word_emb_dict) == 400000, f"Dictionary size is incorrect: should be 400k, but is {len(word_emb_dict)}"
    for key, value in word_emb_dict.items():
        assert value.shape == (200, ), f"Vector shape is incorrect: should be 200 dimensions, but is {value.shape}"
    print("This task looks good!")

check_load_word_emb_dict()
