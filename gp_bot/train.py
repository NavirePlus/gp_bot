"""ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの学習処理の定義."""
from logging import Logger
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

try:
    from gp_bot.bot import GomiPeopleBot
    from gp_bot.config import DEFAULT_PARAMS
    from gp_bot.data import Corpus
    from gp_bot.model import GPLangModel
except:
    from .bot import GomiPeopleBot
    from .config import DEFAULT_PARAMS
    from .data import Corpus
    from .model import GPLangModel


class TrainGPLM(object):
    """ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの学習クラス.

    Attributes
    ----------
    all_total_loss : List[float]
        全epochのLoss
    batch_size : int
        バッチサイズ
    bptt : int
        Back-Propagation Through Time（シーケンスサイズ）
    clip : float
        勾配のNormのClipping閾値
    corpus : :obj:`Corpus`
        コーパス情報
    criterion : torch.nn.CrossEntropyLoss
        Loss関数
    device : torch.device
        CPU・GPUのデバイス情報
    dropout : float
        Dropout率
    emb_size : int
        入力ベクトルの次元数
    epochs : int
        Epoch数
    logger : logging.Logger
        ロガー
    lr : int
        学習率
    model : :obj:`GPLangModel`
        ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデル
    n_hidden : int
        LSTMの隠れユニットの次元数
    n_layers : int
        Reccurentレイヤー数
    n_vocab : int
        語彙数

    """

    def __init__(self, batch_size: int, bptt: int, clip: float, corpus: Corpus, dropout: float, emb_size: int,
                 epochs: int, lr: int, n_hidden: int, n_layers: int,
                 cuda: bool = False, logger: Optional[Logger] = None) -> None:
        """コンストラクタ.

        Parameters
        ----------
        batch_size : int
            バッチサイズ
        bptt : int
            Back-Propagation Through Time（シーケンスサイズ）
        clip : float
            勾配のNormのClipping閾値
        corpus : :obj:`Corpus`
            コーパス情報
        dropout : float
            Dropout率
        emb_size : int
            入力ベクトルの次元数
        epochs : int
            Epoch数
        lr : int
            学習率
        n_hidden : int
            LSTMの隠れユニットの次元数
        n_layers : int
            Reccurentレイヤー数
        cuda : bool, optional
            GPUを利用するか否かのフラグ（デフォルト：利用しない）
        logging : logging.Logger, optional
            ロガー

        """
        self.criterion = nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.bptt = bptt
        self.clip = clip
        self.corpus = corpus
        self.emb_size = emb_size
        self.epochs = epochs
        self.dropout = dropout
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_vocab = len(corpus.dictionary)

        self.all_total_loss: List[float] = []
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = GPLangModel(self.n_vocab, self.emb_size, self.n_hidden,
                                 self.n_layers, self.dropout).to(self.device)
        self.corpus.train_data = self.batchfy(self.corpus.train_data)
        self.logger = logger if logger else GomiPeopleBot.create_logger()

    def batchfy(self, data: Tensor) -> Tensor:
        """バッチ可能なデータに変換する.

        Parameters
        ----------
        data : torch.Tensor
            バッチ化するデータ

        Returns
        -------
        torch.Tensor
            バッチ化したデータ

        """
        n_batch = data.size(0) // self.batch_size
        data = data.narrow(0, 0, n_batch * self.batch_size)
        data = data.view(self.batch_size, -1).t().contiguous()

        return data.to(self.device)

    def dump(self, path: str) -> None:
        """ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルと語彙情報を出力する.

        Parameters
        ----------
        path : str
            モデルファイルパス

        Notes
        -----
        * 語彙情報の出力ファイルはモデルファイルに .vocab を付加したものになる

        """
        with open(path, "wb") as fp:
            torch.save(self.model.state_dict(), fp)  # type: ignore

        self.corpus.dictionary.dump(path + ".vocab")

    def get_batch(self, source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        """指定したindexのバッチデータを取得する.

        Parameters
        ----------
        source : torch.Tensor
            対象データ
        i : int
            バッチのindex

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            時刻TのバッチデータとT+1のバッチデータ

        """
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)

        return data, target

    def repackage_hidden(self, h: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """過去のシーケンスで学習したLSTMの隠れユニット情報を切り離す.

        Parameters
        ----------
        h : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            LSTMの隠れユニット

        Returns
        -------
        Tuple[torch.Tensor, ...]
            新しいLSTMの隠れユニット

        """
        return tuple(v.detach() for v in h)

    def __call__(self, save: Optional[str] = None) -> None:
        """学習開始する.

        Parameters
        ----------
        save : Optional[str], optional
            学習したモデルの出力先

        """
        self.logger.info(f"TrainData: {self.corpus.train_data.size(0)}, Vocabulary: {len(self.corpus.dictionary)}")
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()

            total_loss = 0.0
            t_hidden = self.model.init_hidden(self.batch_size)
            for batch, i in enumerate(range(0, self.corpus.train_data.size(0) - 1, self.bptt)):
                data, targets = self.get_batch(self.corpus.train_data, i)
                self.model.zero_grad()

                t_hidden = self.repackage_hidden(t_hidden)
                res: Tuple[Tensor, Tuple[Tensor, ...]] = self.model(data, t_hidden)
                t_output = res[0]
                t_hidden = res[1]

                loss = self.criterion(t_output.view(-1, self.n_vocab), targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # type: ignore
                for p in self.model.parameters():
                    if p.grad is not None:
                        # https://github.com/pytorch/pytorch/issues/32824
                        p.data.add_(-self.lr, p.grad.data)  # type: ignore

                total_loss += loss.item()

            self.all_total_loss.append(total_loss)

        if save:
            self.dump(save)


if __name__ == "__main__":
    import glob
    import os
    from argparse import ArgumentParser

    torch.manual_seed(0)  # type: ignore

    parser = ArgumentParser(description="ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの学習")
    parser.add_argument("--batch_size", help="バッチサイズ", type=int, default=DEFAULT_PARAMS["BATCH_SIZE"])
    parser.add_argument("--bptt", help="Back-Propagation Through Time（シーケンスサイズ）",
                        type=int, default=DEFAULT_PARAMS["BPTT"])
    parser.add_argument("--clip", help="勾配のNormのClipping閾値", type=float, default=DEFAULT_PARAMS["CLIP"])
    parser.add_argument("--corpus", help="コーパスディレクトリ", type=str, default=DEFAULT_PARAMS["CORPUS"])
    parser.add_argument("--cuda", help="CUDAの利用有無", type=bool, default=DEFAULT_PARAMS["CUDA"])
    parser.add_argument("--dropout", help="Dropout率", type=float, default=DEFAULT_PARAMS["DROPOUT"])
    parser.add_argument("--emb_size", help="入力ベクトルの次元数", type=int, default=DEFAULT_PARAMS["EMB_SIZE"])
    parser.add_argument("--epochs", help="Epoch数", type=int, default=DEFAULT_PARAMS["EPOCHS"])
    parser.add_argument("--lr", help="学習率", type=int, default=DEFAULT_PARAMS["LR"])
    parser.add_argument("--n_hidden", help="LSTMの隠れユニットの次元数", type=int, default=DEFAULT_PARAMS["N_HIDDEN"])
    parser.add_argument("--n_layers", help="Reccurentレイヤー数", type=int, default=DEFAULT_PARAMS["N_LAYERS"])
    parser.add_argument("--save", help="保存するモデルのファイルパス", type=str, default=DEFAULT_PARAMS["SAVE"])
    args = parser.parse_args()

    corpus_files = glob.glob(os.path.join(args.corpus, "*.txt"))
    corpus = Corpus(corpus_files)

    train = TrainGPLM(args.batch_size, args.bptt, args.clip, corpus, args.dropout, args.emb_size, args.epochs,
                      args.lr, args.n_hidden, args.n_layers, args.cuda)
    train(args.save)
    print(train.all_total_loss)
