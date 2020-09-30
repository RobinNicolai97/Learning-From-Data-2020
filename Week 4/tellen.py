words_file = open("named_entity_data.txt", encoding="utf8") #loading a library of bad words to identify toxicity
words_lib = [line.strip().split('\t')[1] for line in words_file.readlines()]

set_lib = set(words_lib)

for word in set_lib:

    print(word, words_lib.count(word)) 