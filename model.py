"""
@author: tsdj

"""


from torch import Tensor, nn


class HISCOClassifier(nn.Module):
    def __init__(self, vocab_size: int = 100, hidden_size: int = 128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1919)

    def forward(self, input_seq: Tensor) -> Tensor:
        out = self.embedding(input_seq)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.classifier(out)

        return out
