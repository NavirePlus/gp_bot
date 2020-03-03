"""APIに関する処理を定義."""
import glob
import os
import traceback
from typing import Dict, List, Optional, Union, no_type_check

from fastapi import FastAPI, HTTPException
from fastapi.logger import logger

from gp_bot.bot import GomiPeopleBot
from gp_bot.config import DEFAULT_PARAMS, N_GENERATE_TEXT, TWEET_MAX_LENGTH
from gp_bot.data import Corpus
from gp_bot.train import TrainGPLM

app = FastAPI()
bot: Optional[GomiPeopleBot] = None


@no_type_check
@app.on_event("startup")
def startup_event():
    """セットアップ処理."""
    if not os.path.exists(DEFAULT_PARAMS["SAVE"]):
        # モデルファイルがない場合は学習する
        corpus_files = glob.glob(os.path.join(DEFAULT_PARAMS["CORPUS"], "*.txt"))
        corpus = Corpus(corpus_files)

        train = TrainGPLM(DEFAULT_PARAMS["BATCH_SIZE"], DEFAULT_PARAMS["BPTT"], DEFAULT_PARAMS["CLIP"], corpus,
                          DEFAULT_PARAMS["DROPOUT"], DEFAULT_PARAMS["EMB_SIZE"], DEFAULT_PARAMS["EPOCHS"],
                          DEFAULT_PARAMS["LR"], DEFAULT_PARAMS["N_HIDDEN"], DEFAULT_PARAMS["N_LAYERS"],
                          DEFAULT_PARAMS["CUDA"], logger)
        train(DEFAULT_PARAMS["SAVE"])

    consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
    consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
    access_token_key = os.environ.get("TWITTER_ACCESS_TOKEN_KEY")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

    # Twiter APIキーが読み込めなかった場合はBotオブジェクトを作らない
    if consumer_key is None or consumer_secret is None or access_token_key is None or access_token_secret is None:
        return

    global bot
    bot = GomiPeopleBot(consumer_key, consumer_secret, access_token_key, access_token_secret,
                        DEFAULT_PARAMS["SAVE"], DEFAULT_PARAMS["CUDA"], logger)


@no_type_check
@app.get("/api/health")
def check_health():
    """ヘルスチェック."""
    return {"status": "ok"}


@no_type_check
@app.post("/api/bot")
def exec_bot(force: bool = False):
    """Botを実行."""
    if bot is None:
        raise HTTPException(status_code=500, detail="Bot is not available.")

    try:
        bot.main(force)

        return {"status": "Successfully bot action."}
    except Exception:
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail="Failed to bot action.")


@no_type_check
@app.post("/api/generate")
def generate_texts(count: int = N_GENERATE_TEXT, max_length: int = TWEET_MAX_LENGTH):
    """ｺﾞﾐﾋﾟｰﾌﾟﾙテキストを生成."""
    try:
        texts: List[Dict[str, Union[str, float]]] = []
        for _ in range(max(1, min(count, 100))):
            text, avg_weight, std_weight = bot.gp_generator(max(1, min(max_length, TWEET_MAX_LENGTH)))

            texts.append({
                "text": text,
                "avg_weight": avg_weight,
                "std_weight": std_weight
            })

        return {"texts": texts}
    except Exception:
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail="Failed to generate Gomi-People texts.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
