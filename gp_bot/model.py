"""ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの定義."""
from typing import Optional, Tuple, no_type_check

import torch.nn as nn
from torch import Tensor


class GPLangModel(nn.Module):  # type: ignore
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
        super(GPLangModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_vocab, n_input)
        self.rnn = nn.LSTM(n_input, n_hidden, n_layers, dropout=dropout)
        self.decoder = nn.Linear(n_hidden, n_vocab)

        self.init_weights()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def init_weights(self) -> None:
        """各層の重みを初期化する."""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

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
        output, t_hidden = self.rnn(emb, t_hidden)
        output = self.drop(output)
        decoded = self.decoder(output)

        return decoded, t_hidden

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
