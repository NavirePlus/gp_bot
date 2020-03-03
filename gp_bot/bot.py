"""ｺﾞﾐﾋﾟｰﾌﾟﾙBotとしての処理定義."""
import json
import os
import random
import traceback
from datetime import datetime
from logging import Logger
from typing import Dict, List, Optional, cast

import numpy as np
import yaml
from twitter import Api

from gp_bot import create_logger
from gp_bot.config import (LAST_MENTION_ID_PATH, MAX_N_MENTIONS, MENTION_MINUTE, N_GENERATE_TEXT, NORMAL_MINUTE,
                           TWEET_MAX_LENGTH)
from gp_bot.generate import GenerateGPText


class GomiPeopleBot(object):
    """ｺﾞﾐﾋﾟｰﾌﾟﾙBotクラス."""

    def __init__(self, consumer_key: str, consumer_secret: str, access_token_key: str, access_token_secret: str,
                 model_filepath: str, cuda: bool, logger: Optional[Logger] = None):
        """コンストラクタ.

        Parameters
        ----------
        consumer_key : str
            TwitterのConsumer Key
        consumer_secret : str
            TwitterのConsumer Secret
        access_token_key : str
            TwitterのAccess Token
        access_token_secret : str
            TwitterのAccess Token Secret
        model_filepath : str
            モデルファイルパス
        cuda : bool
            CUDAの利用するか否かのフラグ
        logger : Optional[logging.Logger], optional
            ロガー

        """
        self.api = Api(consumer_key=consumer_key, consumer_secret=consumer_secret,
                       access_token_key=access_token_key, access_token_secret=access_token_secret)
        self.logger = logger if logger else create_logger()
        self.gp_generator = GenerateGPText(model_filepath, cuda)
        self.steady_tweets = self.load_steady_tweets()

    def load_last_mention_id(self) -> Optional[str]:
        """最後にリプライしたツイートのIDの読み込み.

        Returns
        -------
        Optional[str]
            最後にリプライしたツイートのID

        """
        if not os.path.exists(LAST_MENTION_ID_PATH):
            # ファイルが存在しない場合はNoneを返す
            return None

        with open(LAST_MENTION_ID_PATH) as fp:
            mention_id = json.load(fp)

        return cast(str, mention_id)

    def dump_last_mention_id(self, mention_id: str) -> None:
        """最後にリプライしたツイートのIDの出力.

        Parameters
        ----------
        mention_id : str
            最後にリプライしたツイートのID

        """
        with open(LAST_MENTION_ID_PATH, 'w') as fp:
            json.dump(mention_id, fp)

    def load_steady_tweets(self, path: str = "./steady_tweets.yml") -> List[Dict[str, str]]:
        """定常ツイートを読み込む.

        Parameters
        ----------
        path : str, optional
            定常ツイートを格納したYamlファイルパス（デフォルト：./steady_tweets.yml）

        Returns
        -------
        List[Dict[str, str]]
            定常ツイート情報

        """
        with open(path) as fp:
            return cast(List[Dict[str, str]], yaml.safe_load(fp)["tweets"])

    def generate_text(self, max_length: int) -> str:
        """ｺﾞﾐﾋﾟｰﾌﾟﾙテキストの生成.

        Parameters
        ----------
        max_length : int
            テキストの最大長

        Returns
        -------
        str
            生成したｺﾞﾐﾋﾟｰﾌﾟﾙテキスト

        """
        texts: List[str] = []
        scores: List[float] = []
        for _ in range(N_GENERATE_TEXT):
            text, avg_weight, std_weight = self.gp_generator(max_length)
            texts.append(text)
            # 重みの平均が大きく、分散が小さいほどスコアを高くする
            scores.append(avg_weight / std_weight if std_weight > 0 else 0.0)

        return cast(str, texts[np.argmax(scores)])

    def do_mentions(self, now: datetime, last_mention_id: Optional[str]) -> bool:
        """ユーザへのリプライを行う.

        Parameters
        ----------
        now : datetime.datetime
            現在日時
        last_mention_id : Optional[str]
            最後にリプライしたツイートのID

        Returns
        -------
        bool
            定常ツイートした場合はTrue、しなかった場合はFalse

        Notes
        -----
        * MENTION_MINUTEで指定した分間隔でリプライする

        """
        if int(now.strftime("%M")) % MENTION_MINUTE != 0:
            return False

        random.seed()

        # Botへのリプライを取得
        statuses = self.api.GetMentions(count=100, since_id=last_mention_id)
        if len(statuses) == 0:
            self.logger.info("No reply.")

        if last_mention_id is None and len(statuses) > 0:
            # last_mention_idの指定がなく、リプライがある場合は最新のリプライのツイートIDを格納して終了
            # -> last_mention_idの指定がない場合、過去にリプライしたものに対しても返答してしまうため
            self.dump_last_mention_id(statuses[0].id_str)

            return True

        # リプライをランダムに抽出
        for target in random.sample(statuses, MAX_N_MENTIONS):
            # リプライ対象のツイートごとにテキストを生成してツイート
            text_prefix = f"@{target.user.screen_name} "
            text = self.generate_text(TWEET_MAX_LENGTH - len(text_prefix))

            res = self.post(text_prefix + text, target.id_str)

        # 最後のリプライ（ツイート）IDを記憶
        self.dump_last_mention_id(res)

        return True

    def do_steady_tweets(self, now: datetime) -> bool:
        """定常ツイートを行う.

        Parameters
        ----------
        now : datetime.datetime
            現在日時

        Returns
        -------
        bool
            定常ツイートした場合はTrue、しなかった場合はFalse

        """
        for tweet in self.steady_tweets:
            if now.strftime(tweet["type"]) == tweet["dt"]:
                self.post(tweet["text"])

                return True

        return False

    def post(self, text: Optional[str], mention_id: Optional[str] = None) -> str:
        """ツイート.

        Parameters
        ----------
        text : Optional[str]
            ツイートするテキスト
        mention_id : Optional[str], optional
            リプライ先のツイートID

        Returns
        -------
        str
            ツイートのID

        Raises
        ------
        ValueError
            ツイートするテキストが空の場合

        """
        if text is None or len(text) == 0:
            raise ValueError("Tweet is empty.")

        result = self.api.PostUpdate(status=text, in_reply_to_status_id=mention_id)

        return cast(str, result.id_str)

    def main(self, force: bool) -> None:
        """main関数.

        Parameters
        ----------
        force : bool
            強制的に通常ツイートを行うかどうかのフラグ

        """
        self.logger.info("Start tweet.")

        now = datetime.now()

        try:
            # 定常ツイート
            if not force and self.do_steady_tweets(now):
                # 定常ツイートした場合は終了
                self.logger.info("Successfully steady tweet.")
                return

            # 通常ツイート
            if force or int(now.strftime("%M")) % NORMAL_MINUTE == 0:
                self.post(self.generate_text(TWEET_MAX_LENGTH))
                self.logger.info("Successfully tweet.")
                return

            # リプライ
            last_mention_id = self.load_last_mention_id()
            if self.do_mentions(now, last_mention_id):
                # リプライした場合は終了
                self.logger.info("Successfully replies.")
                return

            self.logger.info("No action.")
        except Exception:
            self.logger.error("Failed to tweet.")
            self.logger.error(traceback.format_exc())
            raise

        return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="ｺﾞﾐﾋﾟｰﾌﾟﾙBotの実行")
    parser.add_argument("--force", help="ランダムツイートを強制的に行うかどうかのフラグ", type=bool, default=False)
    args = parser.parse_args()

    consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
    consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]
    access_token_key = os.environ["TWITTER_ACCESS_TOKEN_KEY"]
    access_token_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
    model_filepath = "./result/model.pth"

    bot = GomiPeopleBot(consumer_key, consumer_secret, access_token_key, access_token_secret,
                        model_filepath, False)
    bot.main(args.force)
