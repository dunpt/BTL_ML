
from transformers import BertTokenizer
from nltk import wordpunct_tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'I will watch Memento tonight'
# bert_input = tokenizer(example_text,padding='max_length', max_length = 7, 
#                        truncation=True, return_tensors="pt")

# print(bert_input['input_ids'])
# print(bert_input['token_type_ids'])
# print(bert_input['attention_mask'])

title = str(example_text)
print(title)
title = " ".join(title.split())

print(wordpunct_tokenize(example_text))