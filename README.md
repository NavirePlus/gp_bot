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
+ Azure Functionsで動いています

## 参考

+ https://github.com/pytorch/examples/blob/master/word_language_model

## Licence

MIT

## Special Thanks

+ [@ピングーBot](http://twitter.com/Pingu_bot)
+ [@くぼみ](http://twitter.com/dekobokoya)

## How to develop

+ Clone this repository and open VSCode with Remote-Containers extension.
