import os

import pytest
from twitter.models import Status


@pytest.fixture(scope="function")
def gp_bot():
    from gp_bot.bot import GomiPeopleBot

    # 単体テスト時に誤って投稿されない（エラーを起こす）ためにあえて _HOGE を付加
    consumer_key = os.environ.get("TWITTER_CONSUMER_KEY") + "_HOGE"
    consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET") + "_HOGE"
    access_token_key = os.environ.get("TWITTER_ACCESS_TOKEN_KEY") + "_HOGE"
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET") + "_HOGE"

    return GomiPeopleBot(consumer_key, consumer_secret, access_token_key, access_token_secret,
                         "/workspaces/gp_bot/result/model.pth")


def mocked_post_tweet(status: str, in_reply_to_status_id: str) -> Status:
    if in_reply_to_status_id is None:
        return Status(id_str="9999_0")
    elif in_reply_to_status_id.startswith("0001"):
        return Status(id_str="9999_1")
    elif in_reply_to_status_id.startswith("0002"):
        return Status(id_str="9999_2")
    elif in_reply_to_status_id.startswith("0003"):
        return Status(id_str="9999_3")
    elif in_reply_to_status_id.startswith("0004"):
        return Status(id_str="9999_4")
    else:
        ValueError(f"Unknown in_reply_to_status_id {in_reply_to_status_id}")
