# Chujun May 8, 2023

# Import libraries
import numpy
from transformers import AlbertTokenizerFast, AlbertModel
import torch

# Define A Lite BERT (ALBERT) 
tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")
model = AlbertModel.from_pretrained("albert-base-v2")

# Make a function that takes in a single line of text and convert it to embedding
def TextToEmbed_OneLine(text):
	inputs = tokenizer(text, return_tensors="pt")
	outputs = model(**inputs)
	# Get the 768-D embeddings for each token 
	token_embedding = outputs.last_hidden_state 
	# Average the embedding acorss tokens to get one average embedding for the sentence
	average_embedding = torch.mean(token_embedding, 1).detach().numpy()
	return average_embedding

# Make a function that takes in the column of the text from the pandas dataframe with multiple lines of text and convert them to embedding
# That is, text is a pandas series with multiple lines of transcripts
# This function also needs to take in the number of features of the embedding which is 768 for ALBERT
def TextToEmbed_Paragraph(text, n_feature):
	# create an empty dataframe to hold the embedding across rows
	# n_feature = 768 for ALBERT
	average_embedding = numpy.empty((0,n_feature),float) 
	# loop over the rows in the text paragraph
	for row in range(text.shape[0]):
		sentence = text.iloc[row]
		inputs = tokenizer(sentence, return_tensors="pt")
		outputs = model(**inputs)
		# Get the 768-D embeddings for each token 
		token_embedding = outputs.last_hidden_state 
		# Average the embedding acorss tokens to get one average embedding for the sentence
		sentence_embedding = torch.mean(token_embedding, 1).detach().numpy()
		# Reshape it the stack in the output
		sentence_embedding_row = numpy.reshape(sentence_embedding, (1,n_feature))
		average_embedding = numpy.append(average_embedding, sentence_embedding_row, axis=0)
	return average_embedding





