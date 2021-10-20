"""ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの学習処理の定義."""
import time
from logging import Logger
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from gp_bot.bot import GomiPeopleBot
    from gp_bot.config import DEFAULT_PARAMS
    from gp_bot.data import Corpus
    from gp_bot.model import GPLangModel, TransformerModel
except:
    from .bot import GomiPeopleBot
    from .config import DEFAULT_PARAMS
    from .data import Corpus
    from .model import GPLangModel, TransformerModel


class TrainGPLM(object):
    """ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルの学習クラス.

    Attributes
    ----------
    all_total_loss: List[float]
        全epochのLoss
    batch_size: int
        バッチサイズ
    bptt: int
        Back-Propagation Through Time（シーケンスサイズ）
    clip: float
        勾配のNormのClipping閾値
    corpus: :obj:`Corpus`
        コーパス情報
    criterion: torch.nn.CrossEntropyLoss
        Loss関数
    device: torch.device
        CPU・GPUのデバイス情報
    dropout: float
        Dropout率
    emb_size: int
        入力ベクトルの次元数
    epochs: int
        Epoch数
    logger: logging.Logger
        ロガー
    lr: int
        学習率
    model: :obj:`GPLangModel`
        ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデル
    n_head: int
        Transformerのヘッド数
    n_hidden: int
        LSTMの隠れユニットの次元数
    n_layers: int
        Reccurentレイヤー数
    n_vocab: int
        語彙数

    """

    def __init__(self, batch_size: int, bptt: int, clip: float, corpus: Corpus, dropout: float, emb_size: int,
                 epochs: int, lr: int, n_head: int, n_hidden: int, n_layers: int,
                 cuda: bool = False, logger: Optional[Logger] = None) -> None:
        """コンストラクタ.

        Parameters
        ----------
        batch_size: int
            バッチサイズ
        bptt: int
            Back-Propagation Through Time（シーケンスサイズ）
        clip: float
            勾配のNormのClipping閾値
        corpus: :obj:`Corpus`
            コーパス情報
        dropout: float
            Dropout率
        emb_size: int
            入力ベクトルの次元数
        epochs: int
            Epoch数
        lr: int
            学習率
        n_head: int
            Transformerのヘッド数
        n_hidden: int
            LSTMの隠れユニットの次元数
        n_layers: int
            Reccurentレイヤー数
        cuda: bool, optional
            GPUを利用するか否かのフラグ（デフォルト：利用しない）
        logging: logging.Logger, optional
            ロガー
        """
        self.batch_size = batch_size
        self.bptt = bptt
        self.clip = clip
        self.corpus = corpus
        self.emb_size = emb_size
        self.n_head = n_head
        self.epochs = epochs
        self.dropout = dropout
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_vocab = len(corpus.dictionary)

        self.device = torch.device("cuda" if cuda else "cpu")
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

    def dump(self, model: TransformerModel, model_dir: str, model_file: str) -> None:
        """ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルと語彙情報を出力する.

        Parameters
        ----------
        path : str
            モデルファイルパス

        Notes
        -----
        * 語彙情報の出力ファイルはモデルファイルに .vocab を付加したものになる

        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, model_file), "wb") as fp:
            torch.save(model, fp)  # type: ignore

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
        if isinstance(h, Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def __call__(self, corpus: Corpus, save_dir: str) -> None:
        """学習開始する.

        Parameters
        ----------
        save: str
            学習したモデルの出力先
        """
        self.logger.info(f"TrainData: {corpus.train_data.size(0)}, Vocabulary: {len(corpus.dictionary)}")
        train_data = self.batchfy(corpus.train_data)

        model = GPLangModel(self.n_vocab, self.emb_size, self.n_hidden, self.n_layers, self.dropout)
        model.to(self.device)

        criterion = nn.NLLLoss()

        best_train_loss = -1.0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            model.train()

            hidden = model.init_hidden(self.batch_size)

            total_loss = 0.0
            for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
                data, targets = self.get_batch(train_data, i)
                model.zero_grad()

                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)

                loss = criterion(output, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                for p in model.parameters():
                    p.data.add_(p.grad, alpha=-self.lr)

                total_loss += loss.item()

            print("-" * 89)
            print("| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f}".format(
                epoch, (time.time() - start_time), total_loss
            ))

            if best_train_loss < 0.0 or total_loss < best_train_loss:
                best_train_loss = total_loss
                print("*** best train loss is updated")
                self.dump(model, save_dir, "best_model.pth")

            print("-" * 89)

        self.dump(model, save_dir, "latest_model.pth")
        corpus.dictionary.dump(os.path.join(save_dir, "latest_model.pth.vocab"))


if __name__ == "__main__":
    import glob
    import os
    import random
    from argparse import ArgumentParser

    random.seed(DEFAULT_PARAMS["RANDOM_SEED"])
    torch.manual_seed(DEFAULT_PARAMS["RANDOM_SEED"])  # type: ignore

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
    parser.add_argument("--n_head", help="Transformerのヘッド数", type=int, default=DEFAULT_PARAMS["N_HEAD"])
    parser.add_argument("--n_hidden", help="LSTMの隠れユニットの次元数", type=int, default=DEFAULT_PARAMS["N_HIDDEN"])
    parser.add_argument("--n_layers", help="Reccurentレイヤー数", type=int, default=DEFAULT_PARAMS["N_LAYERS"])
    parser.add_argument("--save_dir", help="保存するモデルのディレクトリパス", type=str, default=DEFAULT_PARAMS["SAVE_DIR"])
    args = parser.parse_args()

    corpus_files = glob.glob(os.path.join(args.corpus, "*.txt"))
    corpus = Corpus(corpus_files)

    train = TrainGPLM(args.batch_size, args.bptt, args.clip, corpus, args.dropout, args.emb_size, args.epochs,
                      args.lr, args.n_head, args.n_hidden, args.n_layers, args.cuda)
    train(corpus, args.save_dir)
