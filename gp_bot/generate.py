"""ｺﾞﾐﾋﾟｰﾌﾟﾙテキストの生成処理を定義."""
import math
import os
from typing import List, Tuple, cast

import torch
from torch import Tensor

try:
    from gp_bot.data import BOS_SYMBOL, EOS_SYMBOL, Dictionary
    from gp_bot.model import GPLangModel
except:
    from .data import BOS_SYMBOL, EOS_SYMBOL, Dictionary
    from .model import GPLangModel


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

    def __init__(self, temperature: float, model_dir: str, cuda: bool = False) -> None:
        """コンストラクタ.

        Parameters
        ----------
        model_dir : str
            モデルディレトリパス
        cuda : bool, optional
            CUDAを利用するか否かのフラグ（デフォルト：利用しない）

        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.temperature = temperature

        self.vocab = Dictionary()
        self.vocab.load(os.path.join(model_dir, "latest_model.pth.vocab"))

        self.model = self.load_model(model_dir)
        self.model.eval()

    def load_model(self, model_dir: str) -> GPLangModel:
        """モデルディレトリの読み込み.

        Parameters
        ----------
        model_dir : str
            モデルディレトリパス

        Returns
        -------
        :obj:`GPLangModel`
            読み込んだモデル

        """
        with open(os.path.join(model_dir, "best_model.pth"), "rb") as fp:
            model: GPLangModel = torch.load(fp).to(self.device)

        return model

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
        # t_input = torch.randint(len(self.vocab), (1, 1), dtype=torch.long).to(self.device)
        t_hidden = self.model.init_hidden(1)
        with torch.no_grad():
            for _ in range(max_length):
                # 次の文字の語彙IDを予測
                output, t_hidden = self.model(t_input, t_hidden)
                char_weights = output.squeeze().div(self.temperature).exp().cpu()
                char_idx = torch.multinomial(char_weights, 1)[0]
                t_input.fill_(char_idx)

                char = self.vocab.idx2char[int(char_idx.item())]
                weight = char_weights[char_idx]
                if char == BOS_SYMBOL or char == EOS_SYMBOL or weight < 0.01:
                    # 次の文字が開始・終端文字か重みが0.01未満なら終了
                    # -> 重みが極端に小さい場合、不自然な文字の繋がりになるため切る
                    break

                chars += char
                weights.append(weight.item())

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
    parser.add_argument("--temperature", help="", type=float, default=1.0)
    parser.add_argument("--cuda", help="CUDAの利用有無", type=bool, default=False)
    parser.add_argument("--model_dir", help="モデルディレトリパス", type=str, default="./result/")
    args = parser.parse_args()

    gp_generator = GenerateGPText(args.temperature, args.model_dir, args.cuda)
    for _ in range(10):
        text, avg_weight, std_weight = gp_generator(100)
        print(text)
        print(avg_weight)
        print(std_weight)
        try:
            print(avg_weight / std_weight)
        except ZeroDivisionError:
            print("0.0")
        print("-" * 80)
