import os
import time

import azure.functions as func

from gp_bot.bot import GomiPeopleBot


def main(gptimer: func.TimerRequest) -> None:
    consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
    consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
    access_token_key = os.environ.get("TWITTER_ACCESS_TOKEN_KEY")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    bot = GomiPeopleBot(consumer_key, consumer_secret, access_token_key, access_token_secret,
                        "./result/")

    # Azure Functions側の時刻同期のズレを考慮し、5秒遅らせる
    time.sleep(5)

    bot.main()
