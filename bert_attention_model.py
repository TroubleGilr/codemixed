import torch
import torch.nn as nn
from torch.autograd import Variable
#from django.db.models import F,Q
import torch.nn.functional as F

class AttentionModel(torch.nn.Module):
	def __init__(self, bert, batch_size, output_size, hidden_size, vocab_size, embedding_length):
		super(AttentionModel, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM  #LSTM的hidden_​​state的大小
		vocab_size : Size of the vocabulary containing unique words包含唯一单词的词汇表的大小
		embedding_length : Embeddding dimension of GloVe word embeddings  GloVe单词嵌入的嵌入维数
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		        #预训练的GloVe word_embeddings，我们将使用它们来创建word_embedding查找表 
		--------
		
		"""
		
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.bert = bert
		# self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		self.lstm = nn.LSTM(embedding_length, hidden_size)
		self.label = nn.Linear(hidden_size, output_size)
		#self.attn_fc_layer = nn.Linear()
		
	def attention_net(self, lstm_output, final_state):

		""" 
		Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		现在，我们将在我们的LSTM模型中纳入注意机制。在这个新模型中，我们将使用注意力来计算对应的软对齐分数  在LSTM的每个hidden_​​state和最后一个hidden_​​state之间。我们将使用torch.bmm进行批处理矩阵乘法。
		Arguments
		---------
		
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.LSTM的最终输出，其中包含每个序列的隐藏层输出。
		final_state : Final time-step hidden state (h_n) of the LSTM.LSTM的最终时间步隐藏状态（h_n）
		
		---------
		
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  它通过首先计算lstm_output中存在的每个序列的权重，然后最后计算新的隐藏状态。

		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)
					  
		"""
		#使用torch.bmm进行批处理矩阵乘法。
		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		#soft_attn_weights = self.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state
	
	def forward(self, input_sentences, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
		包含pos＆neg类的logit的线性层的输出，它将其输入作为new_hidden_​​state接收，基本上是Attention网络的输出。
		final_output.shape = (batch_size, output_size)
		
		"""
		# print(input_sentences.size())
		input = self.bert(input_sentences)[0]
		input = input.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
		output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
		
		attn_output = self.attention_net(output, final_hidden_state)
		logits = self.label(attn_output)
		
		return logits
