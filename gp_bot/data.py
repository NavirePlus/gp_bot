"""コーパスの読み込み・語彙の管理に関する処理を定義."""
import os
import pickle as pkl
from typing import Dict, List, cast

import torch
from torch import Tensor

# 開始記号と終端記号を定義
BOS_SYMBOL = "<bos>"
EOS_SYMBOL = "<eos>"


class Dictionary(object):
    """文字ベースの語彙（文字種）を管理するクラス.

    Attributes
    ----------
    char2idx : Dict[str, int]
        文字 -> indexのマッピング
    idx2char : List[str]
        index -> 文字のマッピング

    """

    def __init__(self) -> None:
        self.char2idx: Dict[str, int] = {}
        self.idx2char: List[str] = []

    def __len__(self) -> int:
        """語彙の大きさを返す.

        Returns
        -------
        int
            語彙の大きさ

        """
        return len(self.idx2char)

    def add_char(self, char: str) -> None:
        """語彙情報を更新する.

        Parameters
        ----------
        char : str
            語彙として追加する文字

        """
        if char not in self.char2idx:
            self.char2idx[char] = len(self.char2idx)
            self.idx2char.append(char)

    def load(self, path: str) -> None:
        """語彙情報を読み込む.

        Parameters
        ----------
        path : str
            語彙情報が入ったPickleファイルパス

        """
        with open(path, "rb") as fp:
            self.char2idx, self.idx2char = pkl.load(fp)

    def dump(self, path: str) -> None:
        """語彙情報を書き出す.

        Parameters
        ----------
        path : str
            語彙情報を書き出すPickleファイルパス

        """
        with open(path, "wb") as fp:
            pkl.dump((self.char2idx, self.idx2char), fp)


class Corpus(object):
    """コーパスの読み込みを行うクラス.

    Attributes
    ----------
    dictionary : :obj:`Dictionary`
        語彙情報
    train_data : torch.Tensor
        学習データ（文ごとの語彙IDリスト）

    """

    def __init__(self, train_files: List[str]) -> None:
        self.dictionary = Dictionary()
        self.train_data = self.tokenize(train_files)

    def tokenize(self, files: List[str]) -> Tensor:
        """各ファイルを文字列ベースで分割する.

        Parameters
        ----------
        files : List[str]
            読み込むファイルパスのリスト

        Returns
        -------
        torch.Tensor
            文ごとの語彙IDリスト

        """
        ids: List[Tensor] = []
        for path in files:
            # ファイルの有無をチェック
            assert os.path.exists(path)

            with open(path) as fp:
                for line in fp:
                    chars = [BOS_SYMBOL] + list(line.rstrip()) + [EOS_SYMBOL]
                    for c in chars:
                        # 語彙辞書を更新
                        self.dictionary.add_char(c)

                    # 語彙IDのリストをTensorに変換
                    t = torch.tensor([self.dictionary.char2idx[c] for c in chars]).type(torch.int64)
                    ids.append(cast(Tensor, t))

        return torch.cat(ids)
