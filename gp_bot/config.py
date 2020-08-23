"""ｺﾞﾐﾋﾟｰﾌﾟﾙBotに関する設定."""
import os

# 学習のデフォルトパラメータ
DEFAULT_PARAMS = {
    "BATCH_SIZE": 20,
    "BPTT": 50,
    "CLIP": 0.25,
    "CORPUS": "./corpus",
    "CUDA": False,
    "DROPOUT": 0.2,
    "EMB_SIZE": 80,
    "EPOCHS": 80,
    "LR": 20,
    "N_HIDDEN": 80,
    "N_LAYERS": 2,
    "SAVE": "./result/model.pth"
}

# 最大リプライ数（デフォルト：3）
MAX_N_MENTIONS = int(os.environ.get("MAX_N_MENTIONS", "3"))

# リプライの時間間隔（デフォルト：5分）
MENTION_MINUTE = int(os.environ.get("MENTION_MINUTE", "5"))

# 生成するテキストの数（デフォルト：10）
N_GENERATE_TEXT = int(os.environ.get("N_GENERATE_TEXT", "10"))

# 通常ツイートの時間間隔（デフォルト：20分）
NORMAL_MINUTE = int(os.environ.get("NORMAL_MINUTE", "20"))

# ツイートの最大長（デフォルト：140文字）
TWEET_MAX_LENGTH = int(os.environ.get("TWEET_MAX_LENGTH", "140"))
