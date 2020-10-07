import torch
import torch.nn as nn

class FusionNetwork(nn.Module):
    """
    fuse multimodal data into a common representation
    """
    def __init__(self, fuse_method='cat'):
        super(FusionNetwork, self).__init__()
        self._fuse_method = fuse_method.lower()

    def forward(self, hidden, input):    #[2,3,200], [3,4,400], [3,4,400]
        """
        input_a and input_b have a same dimension
        :param input_a: [batch, length, dim]
        :param input_b: [batch, length, dim]
        :return:
        """
        # relation_len = input_a.size(1)*input_b.size(1)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)                        # [3, 400]
        a_ext = input.unsqueeze(2).expand(-1, -1, input.size(1), -1)  # [3, 4, 4, 400]
        b_ext = input.unsqueeze(1).expand(-1, input.size(1), -1, -1)  # [3, 4, 4, 400]
        h_ext = hidden.unsqueeze(1).unsqueeze(2).expand(-1, input.size(1), input.size(1), -1)  # [3, 4, 4, 400]
        fused = None
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if 'cat' in self._fuse_method:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            fused = torch.cat([a_ext, b_ext, h_ext], dim=-1).mean(2)
        elif 'add' in self._fuse_method:
            fused = torch.add(torch.add(a_ext, b_ext), h_ext).mean(2)
        elif 'mul' in self._fuse_method:
            fused = torch.mul(torch.mul(a_ext, b_ext), h_ext).mean(2)
        return fused

class RelationNetwork(nn.Module):
    def __init__(self, dim_relation, fuse_method='cat'):
        super(RelationNetwork, self).__init__()
        if 'cat' in fuse_method.lower():
            input_dim = 3*dim_relation
        else:
            input_dim = dim_relation
        # for relation output
        self._relation = nn.Sequential(
            nn.Linear(input_dim, dim_relation),
            nn.ReLU()
        )

    def forward(self, input_fused):     # [3,16,1200]
        relation = self._relation(input_fused)
        return relation

class FuseRelationNetwork(nn.Module):
    """
    fuse multimodal data and generate their relations
    """
    def __init__(self, dim_relation, fuse_method='cat'):
        super(FuseRelationNetwork, self).__init__()
        self._fuse = FusionNetwork(fuse_method=fuse_method)
        self._relation = RelationNetwork(dim_relation=dim_relation, fuse_method=fuse_method)

    def forward(self,  hidden, input):    # [2,3,200], [3,4,400]
        # get crossing fused representation
        input_fused = self._fuse(hidden, input)  #[9*16, 400]
        # get relation scores based on the crossing fused representation
        relation = self._relation(input_fused)
        return relation


if __name__ == '__main__':
    embedding_dim = 128
    hidden_size = 200
    src_vocab_size = 30
    tgt_vocab_size = 40
    embedding_layer = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embedding_dim)
    lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
    RN = FuseRelationNetwork(hidden_size*2)

    a = torch.tensor([[1,2,3,4],[5,6,7,0],[8,9,0,0]])
    emb_a = embedding_layer(a)
    input_a = nn.utils.rnn.pack_padded_sequence(emb_a, [4,3,2], batch_first=True)
    encoding_a,(h,_) = lstm(input_a)
    encoding_a, _ = nn.utils.rnn.pad_packed_sequence(encoding_a, batch_first=True)
    r = RN(h, encoding_a)
    print(1)

