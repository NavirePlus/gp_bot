from twitter.models import Status, User


MENTIONED_TWEETS = [
    Status(id_str="0001_1", user=User(screen_name="user0001")),
    Status(id_str="0001_2", user=User(screen_name="user0001")),
    Status(id_str="0002_1", user=User(screen_name="user0002")),
    Status(id_str="0003_1", user=User(screen_name="user0003")),
    Status(id_str="0004_1", user=User(screen_name="user0004")),
]

SELF_TWEETS = [
    Status(id_str="9999_1"),
    Status(id_str="9999_2"),
    Status(id_str="9999_3", in_reply_to_status_id="0003_1"),
    Status(id_str="9999_4")
]
