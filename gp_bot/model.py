"""ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの定義."""
from typing import Optional, Tuple, no_type_check

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GPLangModel(nn.Module):
    """ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルのクラス.

    Attributes
    ----------
    decoder : torch.nn.Linear
        Decoder層
    drop : torch.nn.Dropout
        Dropout
    encoder : torch.nn.Embedding
        Encoder層
    n_hidden : int
        LSTMの隠れユニットの次元数
    n_input : int
        入力ベクトルの次元数
    rnn : torch.nn.LSTM
        LSMT層

    """

    def __init__(self, n_vocab: int, n_input: int, n_hidden: int, n_layers: int, dropout: float) -> None:
        """コンストラクタ.

        Parameters
        ----------
        n_vocab : int
            語彙数
        n_input : int
            入力ベクトルの次元数
        n_hidden : int
            LSTMの隠れユニットの次元数
        n_layers : int
            Reccurentレイヤー数
        dropout : float
            Dropout率

        """
        super(GPLangModel, self).__init__()  # type: ignore

        self.n_vocab = n_vocab
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_vocab, n_input)
        self.rnn = nn.LSTM(n_input, n_hidden, n_layers, dropout=dropout)  # type: ignore
        self.decoder = nn.Linear(n_hidden, n_vocab)

        self.init_weights()

    def init_weights(self) -> None:
        """各層の重みを初期化する."""
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    @no_type_check
    def forward(self, t_input: Tensor, t_hidden: Optional[Tuple[Tensor, Tensor]]) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """順伝播関数.

        Parameters
        ----------
        t_input : torch.Tensor
            入力ベクトル
        t_hidden : Optional[Tuple[torch.Tensor, torch.Tensor]]
            LSTMの隠れユニットの特徴ベクトル

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            デコード済みベクトルと更新済み隠れユニットの特徴ベクトル

        """
        emb = self.drop(self.encoder(t_input))
        output, hidden = self.rnn(emb, t_hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.n_vocab)

        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, ...]:
        """隠れユニットの特徴ベクトルを初期化する.

        Parameters
        ----------
        batch_size : int
            バッチサイズ

        Returns
        -------
        Tuple[torch.Tensor, ...]
            LSTMの隠れユニットの特徴ベクトル

        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                weight.new_zeros(self.n_layers, batch_size, self.n_hidden))
