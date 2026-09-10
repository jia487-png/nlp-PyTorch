# nlp-PyTorch
神经网络工程包
# pytorch_Lstm.py
~~~~
# Author: Robert Guthrie
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
lstm=torch.nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
torch.manual_seed(1)
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
~~~~
# 运行结果
tensor([[[-0.0069,  0.0047,  0.3768]],  

        [[-0.0713,  0.0829,  0.1027]],  

        [[-0.0010, -0.0897, -0.0720]],  

        [[-0.0380, -0.0021, -0.0923]],  

        [[-0.0820,  0.0991, -0.1158]]], grad_fn=<MkldnnRnnLayerBackward0>)   
(tensor([[[-0.0820,  0.0991, -0.1158]]], grad_fn=<StackBackward0>), tensor([[[-0.2937,  0.2913, -0.2318]]], grad_fn=<StackBackward0>))  



# pytorch_LSTM.py
~~~~
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("the cat ate the fish".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody Sing the song".split(), ["NN", "V", "DET", "NN"]),
]

word_to_ix = {}
for sent, _ in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    total_loss = 0
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch:03d} loss: {total_loss:.4f}")

with torch.no_grad():
    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        predictions = model(sentence_in)
        pred_tags = [list(tag_to_ix.keys())[torch.argmax(pred).item()] for pred in predictions]
        print(sentence, "->", pred_tags)
        print("真实标签:", tags)
~~~~

# 运行结果
epoch 000 loss: 2.3192
epoch 050 loss: 1.6975
epoch 100 loss: 0.8788
epoch 150 loss: 0.2744
epoch 200 loss: 0.1303
epoch 250 loss: 0.0796
['the', 'cat', 'ate', 'the', 'fish'] -> ['DET', 'NN', 'V', 'DET', 'NN']
真实标签: ['DET', 'NN', 'V', 'DET', 'NN']
['Everybody', 'Sing', 'the', 'song'] -> ['NN', 'V', 'DET', 'NN']
真实标签: ['NN', 'V', 'DET', 'NN']
