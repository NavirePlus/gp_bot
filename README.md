# Gomi People Bot

## Gomi Peolpleとは

1. 半角ｶﾅを愛し
1. 同族を疎まず
1. 不正ｻﾝｼﾞﾊﾝをしない
1. ﾐｬｯﾐｬｰﾑｫﾑｫｯﾂｧﾚﾗﾎﾟﾎﾟ

## Gomi People Bot

+ ｺﾞﾐﾋﾟｰﾌﾟﾙ言語モデルをDeep Learningで学習し、ｺﾞﾐﾋﾟｰﾌﾟﾙテキストを生成、Twitterに投稿するBotです
    + 内部はLSTMを使ったシンプルなEncoder-Decoderモデルです
+ リプライを5分に1回、ランダムにｺﾞﾐﾋﾟｰﾌﾟﾙテキストを20分に1回呟きます

## 利用方法

### 前提条件

+ Docker 19.03+
+ docker-compose 1.25.0+

### 起動

```
$ cd /path/to
$ git clone https://github.com/NavirePlus/gp_bot.git
$ cd gp_bot/
$ docker-compose build
$ docker-compose up -d
```
+ `.env`ファイルおよびファイル中の4つの環境変数を定義していないと起動に失敗します
+ デフォルトで8080ポートでAPIサーバが立ちます
    + ポートを帰る場合は`docker-compose.yml`の`ports`を変更してください
+ APIサーバ起動時、学習済みのモデルファイルがない場合、学習が自動的に始まります
    + マシンスペックにもよりますが、数分で学習は終わります

### （オプション）TwitterのBotとして動かす場合

+ Twitterのbotとして動かす場合、TwitterのAPIキーの取得が必要です
    + 「Twitter consumer key」 などとググればやり方いっぱい出てきます

```
$ cd /path/to/gp_bot/
$ cp .env.sample .env
$ vi .env
TWITTER_CONSUMER_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX         <- Consumer Keyを指定
TWITTER_CONSUMER_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX      <- Consumer Secretを指定
TWITTER_ACCESS_TOKEN_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX     <- Access Token Keyを指定
TWITTER_ACCESS_TOKEN_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  <- Access Token Secretを指定
$ docker-compose restart
```

### API

#### ｺﾞﾐﾋﾟｰﾌﾟﾙテキスト生成

```
$ curl -s -XPOST localhost:8080/api/generate?count=10&max_length=140
{
    "texts": [
        {
            "text": "...",      // 生成されたテキスト
            "avg_weight": X.Y,  // テキスト生成時の各文字の重みの平均
            "std_weight": Y.Z   // テキスト生成時の各文字の重みの分散
        },
        ...
    ]
}
```
+ デフォルトで10件生成されます
    + 生成件数を変えたい場合はクエリパラメータ`count`の値を変更します（指定可能範囲は1～100）
    + 指定可能範囲外の値の場合は1件または100件生成されます
+ デフォルトで最大140文字のテキストが生成されます
    + 最大文字数を変えたい場合はクエリパラメータ`max_length`の値を変更します（指定可能範囲は1～140）
    + 指定可能範囲外の値の場合は1文字または140文字が最大長になります

#### Bot実行

```
$ curl -s -XPOST localhost:8080/api/bot?force=false
{"status": "Successfully bot action."}
```
+ リプライを5分に1回、ランダムにｺﾞﾐﾋﾟｰﾌﾟﾙテキストを20分に1回呟きます
    + その他、`steady_tweets.yml`に定義した日時に従って固定ツイートを呟きます
+ 時刻に関係なくランダムのツイートを行う場合は、クエリパラメータ`force`を`true`にします
+ TwitterのAPIキーの指定がなかったり間違っている場合はつぶやきに失敗します

## 停止

```
$ cd /path/to/gp_bot/
$ docker-compose stop
```

## 削除

```
$ cd /path/to/gp_bot/
$ docker-compose down
$ docker rmi gp_bot
```

## 参考

+ https://github.com/pytorch/examples/blob/master/word_language_model

## Licence

MIT

## Special Thanks

+ 今は亡き[ピングーBot](http://twitter.com/Pingu_bot)の生みの親である[くぼみ](http://twitter.com/dekobokoya)

## 開発手順

+ このリポジトリをcloneした後、[VSCodeのRemote-Containers](https://code.visualstudio.com/docs/remote/containers)を使うと自動的に開発環境がコンテナで構築されます。
+ Remote-Containersを使うためにはお使いの環境にDockerおよびdocker-composeがインストールされている必要があります。
    + WSL2 + Docker Desctop for Windowsでも可能です

## TODO

+ [x] Add testing codes
