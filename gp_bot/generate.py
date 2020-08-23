"""ｺﾞﾐﾋﾟｰﾌﾟﾙテキストの生成処理を定義."""
import math
from typing import List, Tuple, cast, no_type_check

import torch
from torch import Tensor

from gp_bot.data import BOS_SYMBOL, EOS_SYMBOL, Dictionary
from gp_bot.model import GPLangModel


class GenerateGPText(object):
    """ｺﾞﾐﾋﾟｰﾌﾟﾙテキストの生成クラス.

    Attributes
    ----------
    device : torch.device
        CPU・GPUのデバイス情報
    model : :obj:`GPLangModel`
        学習済みのモデル
    vocab : :obj:`Dictionary`
        語彙情報

    """

    def __init__(self, model_file: str, cuda: bool = False) -> None:
        """コンストラクタ.

        Parameters
        ----------
        model_file : str
            モデルファイルパス
        cuda : bool, optional
            CUDAを利用するか否かのフラグ（デフォルト：利用しない）

        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = self.load_model(model_file)
        self.model.eval()

        self.vocab = Dictionary()
        self.vocab.load(model_file + ".vocab")

    @no_type_check
    def load_model(self, path: str) -> GPLangModel:
        """モデルファイルの読み込み.

        Parameters
        ----------
        path : str
            モデルファイルパス

        Returns
        -------
        :obj:`GPLangModel`
            読み込んだモデル

        """
        with open(path, "rb") as fp:
            return torch.load(fp).to(self.device)

    def __call__(self, max_length: int) -> Tuple[str, float, float]:
        """テキストの生成.

        Parameters
        ----------
        max_length : int
            最大テキスト長

        Returns
        -------
        Tuple[str, float, float]
            生成したテキスト、重みの平均、重みの分散

        """
        chars = ""
        weights: List[float] = []

        # 入力ベクトルとLSTMの隠れユニットの特徴ベクトルを初期化
        t_input = cast(Tensor, torch.tensor([[self.vocab.char2idx[BOS_SYMBOL]]]  # type: ignore
                                            ).to(self.device).type(torch.long))
        t_hidden = self.model.init_hidden(1)
        with torch.no_grad():
            for _ in range(max_length):
                res: Tuple[Tensor, Tuple[Tensor, ...]] = self.model(t_input, t_hidden)
                output = res[0]
                t_hidden = res[1]

                # 次の文字の語彙IDを予測
                char_weights: Tensor = torch.div(output.squeeze(), 1.0).exp().cpu()
                char_idx = torch.multinomial(char_weights, 1)[0]

                char = self.vocab.idx2char[int(char_idx.item())]
                weight = char_weights.log()[char_idx]
                if char == BOS_SYMBOL or char == EOS_SYMBOL or weight < 1.0:
                    # 次の文字が開始・終端文字か重みが1未満なら終了
                    # -> 重みが極端に小さい場合、不自然な文字の繋がりになるため切る
                    break

                chars += char
                weights.append(weight.item())

                t_input.fill_(char_idx)

        if len(weights) == 0 or len(chars.strip()) == 0:
            # 1文字も生成されなかった場合は再生成する
            return self.__call__(max_length)

        t_weights = torch.tensor(weights)

        avg_weight = t_weights.mean().item()
        std_weight = t_weights.std().item()
        if math.isnan(std_weight):
            std_weight = 0.0

        return chars, avg_weight, std_weight


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="ｺﾞﾐﾋﾟｰﾌﾟﾙテキストの生成")
    parser.add_argument("--cuda", help="CUDAの利用有無", type=bool, default=False)
    parser.add_argument("--model", help="モデルファイルパス", type=str, default="./result/model.pth")
    args = parser.parse_args()

    gp_generator = GenerateGPText(args.model, args.cuda)
    for _ in range(10):
        text, avg_weight, std_weight = gp_generator(100)
        print(text)
        print(avg_weight)
        print(std_weight)
        print(avg_weight / std_weight)
        print("-" * 80)
