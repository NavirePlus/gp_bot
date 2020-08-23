import datetime
import logging
import os

import azure.functions as func

from gp_bot.bot import GomiPeopleBot


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)

    consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
    consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
    access_token_key = os.environ.get("TWITTER_ACCESS_TOKEN_KEY")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    bot = GomiPeopleBot(consumer_key, consumer_secret, access_token_key, access_token_secret,
                        "./result/model.pth")
    bot.main(force=True)
