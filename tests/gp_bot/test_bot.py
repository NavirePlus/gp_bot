from datetime import datetime

import pytest
from pytest_mock import MockerFixture
from twitter import Api as TwitterAPI

from gp_bot.bot import GomiPeopleBot
from gp_bot.config import MAX_N_MENTIONS

from tests.conftest import mocked_post_tweet
from tests.data.tweets import MENTIONED_TWEETS, SELF_TWEETS


class Test_GomiPeopleBot(object):

    def test_get_latest_status_id(self, mocker: MockerFixture, gp_bot: GomiPeopleBot):
        # 自身のツイートが存在しない場合
        mocker.patch.object(TwitterAPI, "GetUserTimeline", return_value=[])
        assert gp_bot.get_latest_status_id() is None

        # 自身のツイートが存在するが、リプライツイートがない場合
        mocker.patch.object(TwitterAPI, "GetUserTimeline", return_value=SELF_TWEETS[:2])
        assert gp_bot.get_latest_status_id() == "9999_1"

        # 自身のツイートが存在し、リプライツイートがある場合
        mocker.patch.object(TwitterAPI, "GetUserTimeline", return_value=SELF_TWEETS)
        assert gp_bot.get_latest_status_id() == "9999_3"

    def test_load_steady_tweets(self, gp_bot: GomiPeopleBot):
        tweets = gp_bot.load_steady_tweets("./steady_tweets.yml")

        # 1件だけ確認する
        assert {"type": "%H:%M", "dt": "03:30", "text": "ｻﾝｼﾞﾊﾝ!!"} in tweets

    def test_generate_text(self, gp_bot: GomiPeopleBot):
        text = gp_bot.generate_text(20)
        assert isinstance(text, str)
        assert len(text) <= 20

    def test_do_mentions(self, mocker: MockerFixture, gp_bot: GomiPeopleBot):
        # 指定した分間隔でない場合
        assert gp_bot.do_mentions(datetime(2020, 1, 1, 0, 8, 0), "TEST_MENTION_ID") == False

        # メンションが1件もない場合
        mocker.patch.object(TwitterAPI, "GetMentions", return_value=[])
        assert gp_bot.do_mentions(datetime(2020, 1, 1, 0, 20, 0), "TEST_MENTION_ID") == False

        # last_mention_idがNoneかつメンションがある場合
        mocker.patch.object(TwitterAPI, "GetMentions", return_value=MENTIONED_TWEETS)
        assert gp_bot.do_mentions(datetime(2020, 1, 1, 0, 20, 0), None) == False

        # 返信を行う場合で受けたメンションがMAX_N_MENTIONSで指定された件数以下の場合
        mocker.patch.object(TwitterAPI, "PostUpdate", side_effect=mocked_post_tweet)
        mocker.patch.object(TwitterAPI, "GetMentions", return_value=MENTIONED_TWEETS[:4])
        assert gp_bot.do_mentions(datetime(2020, 1, 1, 0, 20, 0), "TEST_MENTION_ID") == True

        # 返信を行う場合で受けたメンションがMAX_N_MENTIONSで指定された件数より多い場合
        posted_mock = mocker.patch.object(TwitterAPI, "PostUpdate", side_effect=mocked_post_tweet)
        mocker.patch.object(TwitterAPI, "GetMentions", return_value=MENTIONED_TWEETS)
        assert gp_bot.do_mentions(datetime(2020, 1, 1, 0, 20, 0), "TEST_MENTION_ID") == True
        # メンション先はランダムで決まるため、PostUpdateのモック関数が呼ばれた回数で判断
        assert posted_mock.call_count == MAX_N_MENTIONS

    def test_do_steady_tweets(self, mocker: MockerFixture, gp_bot: GomiPeopleBot):
        # 静的ツイートに該当しない時間の場合
        assert gp_bot.do_steady_tweets(datetime(2020, 1, 1, 0, 8, 0)) == False

        # 静的ツイートに該当する時間の場合（サンプルで定時報告のみ）
        mocker.patch.object(TwitterAPI, "PostUpdate", side_effect=mocked_post_tweet)
        assert gp_bot.do_steady_tweets(datetime(2020, 1, 1, 3, 30, 0)) == True

    def test_post(self, mocker: MockerFixture, gp_bot: GomiPeopleBot):
        mocker.patch.object(TwitterAPI, "PostUpdate", side_effect=mocked_post_tweet)

        # メンションIDありの場合
        assert gp_bot.post("test", "0001_1") == "9999_1"

        # メンションIDなし（None）の場合
        assert gp_bot.post("test") == "9999_0"

        # テキストがNoneの場合
        with pytest.raises(ValueError):
            gp_bot.post(None)

        # テキストが空文字の場合
        with pytest.raises(ValueError):
            gp_bot.post("")
