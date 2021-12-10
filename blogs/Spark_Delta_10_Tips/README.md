# Spark/Deltaを使ったデータ分析・処理の書き方 - 10のTips

こんにちは、Databricksの北村です。
今回は、データ分析・処理でSparkとDelta Lakeを使う際によく使うTipsを見ていきたいと思います。実際に、あるCSVファイルがあるときに、それをどのようにSparkのコードに落としていくかを順を追って見ていきたいと思います。

今回の記事では、Databricksの独自機能はあまり焦点を当てておらず、一般的なSparkとDelta Lakeの環境でも全く同様に動作します(もちろんDatabricks上でもそのまま動きます)。

Sparkとかよく聞くけど、実際どんな感じで使えるのかな、という方向けへの視点も取り入れておりますので、ぜひ読んでみてください。

## Tips 0. 使用するデータ


今回は、CSVデータとしてDatabricksのサンプルデータセットに含まれる`diamonds.csv`を使っていきます。Databricks上では`dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv`でアクセスできます。

[ggplot2のGithub上](https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv)で公開されているデータになりますので、Databricks環境以外で試したい場合は、ダウンロードしてお使いください。その際は、サンプルコード内のファイルパスも適宜置換してください。また、github上のデータはDatabricks版のデータとは異なり、最初のカラム`_c0`がないものになっております。


## Tips 1. とりあえず、CSVを読み込む

CSVファイルが用意できたので、とりあえずSpark上に読み込んでみましょう。
(Sparkもpandas同様、データフレームでデータを扱えます)

```
>>> df = spark.read.csv('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
>>> df.show(5)

+----+-----+-------+-----+-------+-----+-----+-----+----+----+----+
| _c0|  _c1|    _c2|  _c3|    _c4|  _c5|  _c6|  _c7| _c8| _c9|_c10|
+----+-----+-------+-----+-------+-----+-----+-----+----+----+----+
|null|carat|    cut|color|clarity|depth|table|price|   x|   y|   z|
|   1| 0.23|  Ideal|    E|    SI2| 61.5|   55|  326|3.95|3.98|2.43|
|   2| 0.21|Premium|    E|    SI1| 59.8|   61|  326|3.89|3.84|2.31|
|   3| 0.23|   Good|    E|    VS1| 56.9|   65|  327|4.05|4.07|2.31|
|   4| 0.29|Premium|    I|    VS2| 62.4|   58|  334| 4.2|4.23|2.63|
+----+-----+-------+-----+-------+-----+-----+-----+----+----+----+
only showing top 5 rows
```

(注: 上記の結果は通常のpysparkのインタプリタで実行した場合の表示形式になります。ただし、煩わしいので、以降はコード部分と結果表示部分がわかるように分離してシンプルな形式で記載していきます。)

おやおや、結果をみると何かおかしいですね。一行目がヘッダの内容になっているのがわかります。実際にCSVファイルを見てみると以下のようになっています。

```csv
"","carat","cut","color","clarity","depth","table","price","x","y","z"
"1",0.23,"Ideal","E","SI2",61.5,55,326,3.95,3.98,2.43
"2",0.21,"Premium","E","SI1",59.8,61,326,3.89,3.84,2.31
"3",0.23,"Good","E","VS1",56.9,65,327,4.05,4.07,2.31
"4",0.29,"Premium","I","VS2",62.4,58,334,4.2,4.23,2.63
...
```

Sparkで読み込むときにCSVの一行目はHeader情報というオプションを指定してみます。

```
df = spark.read.csv('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv', header=True)
df.show(5)

[結果]
+---+-----+-------+-----+-------+-----+-----+-----+----+----+----+
|_c0|carat|    cut|color|clarity|depth|table|price|   x|   y|   z|
+---+-----+-------+-----+-------+-----+-----+-----+----+----+----+
|  1| 0.23|  Ideal|    E|    SI2| 61.5|   55|  326|3.95|3.98|2.43|
|  2| 0.21|Premium|    E|    SI1| 59.8|   61|  326|3.89|3.84|2.31|
|  3| 0.23|   Good|    E|    VS1| 56.9|   65|  327|4.05|4.07|2.31|
|  4| 0.29|Premium|    I|    VS2| 62.4|   58|  334| 4.2|4.23|2.63|
|  5| 0.31|   Good|    J|    SI2| 63.3|   58|  335|4.34|4.35|2.75|
+---+-----+-------+-----+-------+-----+-----+-----+----+----+----+
only showing top 5 rows
```

今度はうまく読み込めました。

## Tips 2. 読み込むコードを一般化する

実は、Sparkでファイルを読み込むコードはいくつかの書き方があります。
上記で使った`spark.read.csv()`のコードは以下の書き方で置き換えられます。

```
df = spark.read.format('csv').option('header', True).load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
df.show(5)
```

好みの問題ではあると思いますが、個人的には上記の書き方の方が
 * 一般化されている
 * コードからファイル形式、オプション指定、ファイルパスが明示的になっているので見やすい
という理由で、望ましいと思っています(あくまで個人の感想です)。以降はこの形式で書いていきます。


今回扱うファイルはCSVですが、Sparkはもちろん他のファイル、例えばJSON, Parquet, Avro, text, 画像データなども読み込めます。それではJSONファイルを読み込む場合はどのようなコードになるのでしょうか?そうです、`format('json')`になります。

## Tips 3. 可読性を上げる

上記のコードの書き方だと、一行が非常に長くなってします。Pythonの文法では`()`内ではドット(`.`)始まりの場合に改行が許されているので、以下のように書き直せます。

```
df = (
  spark.read
  .format('csv')
  .option('header',True)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.show(5)
```

読み込むフォーマット、オプション、ファイルパスがより分離して見やすい形になっており、可読性が向上します。


## Tips 4. スキーマ(各カラムのデータ型)を確認する・整える

読み込んだ結果の表示をみる限り、うまくCSVファイルのデータが読み込まれているように見えます。ただし、コードの中にはデータ型やスキーマを指定していません。実際にどのようなデータ型で識別されているのでしょうか?見てみましょう。

```
df.printSchema()

[結果]
root
 |-- _c0: string (nullable = true)
 |-- carat: string (nullable = true)
 |-- cut: string (nullable = true)
 |-- color: string (nullable = true)
 |-- clarity: string (nullable = true)
 |-- depth: string (nullable = true)
 |-- table: string (nullable = true)
 |-- price: string (nullable = true)
 |-- x: string (nullable = true)
 |-- y: string (nullable = true)
 |-- z: string (nullable = true)
```

全て文字列型で読み込まれていました。文字列型では数値としての大小比較・演算ができませんので、このままでは分析できません。
Sparkにはデータをファイルから読み込む際にスキーマを推定する機能(`inferSchema`)があります。これを使ってみましょう。

```
df = (
  spark.read
  .format('csv')
  .option('header', True)
  .option('inferSchema', True)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.printSchema()

[結果]
root
 |-- _c0: integer (nullable = true)
 |-- carat: double (nullable = true)
 |-- cut: string (nullable = true)
 |-- color: string (nullable = true)
 |-- clarity: string (nullable = true)
 |-- depth: double (nullable = true)
 |-- table: double (nullable = true)
 |-- price: integer (nullable = true)
 |-- x: double (nullable = true)
 |-- y: double (nullable = true)
 |-- z: double (nullable = true)
```

コードのどの部分が変わったかわかりますか?そうです、`.option('inferSchema', True)`の部分です。この書き方だと一目瞭然ですね。

そして、スキーマの結果もそれぞれのカラムの内容に沿ったものになっているようです。

## Tips 5. スキーマをさらに整える

もちろんスキーマはコード上で明示的に指定して、ファイルを読み込むこともできます。
スキーマの書き方は以下の2通りあります。

* スキーマの書き方1: 一般的なDDL形式で書く 
* スキーマの書き方2: PySparkのスキーマを定義しているClassで書く

どちらも一長一短ですが、一般的にカジュアルに書くなら前者、きっちり書くなら後者の形式かと思います。
せっかくなので、両方みていきますので、比較してみてください。

### 一般的なDDL形式で書く

まず、コードをみてみましょう。

```
schema_DDLformat = '''
_c0 Integer,
carat DOUBLE,
cut String,
color String,
clarity String,
depth DOUBLE,
table Integer,
price Integer,
x DOUBLE,
y DOUBLE,
z DOUBLE
'''

df = (
  spark.read
  .format('csv')
  .option('header',True)
  .schema(schema_DDLformat)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.printSchema()

[結果]
root
 |-- _c0: integer (nullable = true)
 |-- carat: double (nullable = true)
 |-- cut: string (nullable = true)
 |-- color: string (nullable = true)
 |-- clarity: string (nullable = true)
 |-- depth: double (nullable = true)
 |-- table: integer (nullable = true)
 |-- price: integer (nullable = true)
 |-- x: double (nullable = true)
 |-- y: double (nullable = true)
 |-- z: double (nullable = true)
```

最初の部分でスキーマをDDL形式で書いて、`schema_DDLformat`変数に入れています。
その後、ファイルを読み込む際に`schema()`関数でスキーマを渡しています。
DDLなので、大文字・小文字の区別はありません。全て小文字で書いてもコードの実行結果は変わりません。

スキーマ表示の結果も指定した通りになっていることがわかります。


### PySparkのスキーマを定義しているClassで書く

もちろんPySparkの中ではスキーマは`class StructType`として定義されています。それを使ったスキーマの指定方法も見ていきます。
こちらもコードから見てみましょう。

```
from pyspark.sql.types import *

schema_StructType = StructType([
  StructField('_c0', IntegerType(), True),
  StructField('carat', DoubleType(), True),
  StructField('cut', StringType(), True),
  StructField('color', StringType(), True),
  StructField('clarity', StringType(), True),
  StructField('depth', DoubleType(), True),
  StructField('table', IntegerType(), True),
  StructField('price', IntegerType(), True),
  StructField('x', DoubleType(), True),
  StructField('y', DoubleType(), True),
  StructField('z', DoubleType(), True)
])

df = (
  spark.read
  .format('csv')
  .option('header',True)
  .schema(schema_StructType)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.printSchema()

[結果]
root
 |-- _c0: integer (nullable = true)
 |-- carat: double (nullable = true)
 |-- cut: string (nullable = true)
 |-- color: string (nullable = true)
 |-- clarity: string (nullable = true)
 |-- depth: double (nullable = true)
 |-- table: integer (nullable = true)
 |-- price: integer (nullable = true)
 |-- x: double (nullable = true)
 |-- y: double (nullable = true)
 |-- z: double (nullable = true)
```

DDLの書き方と比較するとわかりやすいと思います。
`class StructType`のコンストラクタに配列としてスキーマを指定します。
一つのカラム(データ型)は`class StructField`で表現されます。
より「きっちり」感が出ていると思います。


## Tips 6. スキーマのコードをcheatする

Sparkでのデータ処理を書く場合、データ品質の側面を考えると、ほとんどのケースでユーザーがスキーマを指定することになると思います。一方で、カラム数が膨大なデータに対して、スクラッチからスキーマをコードで書くのは骨が折れます。そこで、cheatの一例をお見せします(正攻法とは言い難いので、場合に応じて使ってみてください)。

DDLでスキーマを書く方法をみると、実は`printSchema()`の表示結果とほぼ似ていることに気づきます。そのため、読み込むデータの一部分を用意して、最初に`inferSchema`を使用して読み込み、`printSchema()`結果から文字列置換してDDL形式のスキーマコードを作成することができます。必要な部分(データ型部分)のみを変更すれば良いので、テストでコードを書く際にはよく使います。

また、`class StructType`でスキーマを指定する場合についても同様のcheatがあります。まず、先ほどと同様に`inferSchema`でデータフレームに一部のデータを読み込んでおきます。データフレームは属性`schema`にスキーマを持っているので、それを使います。

```
df_pre = (
  spark.read
  .format('csv')
  .option('header', True)
  .option('inferSchema', True)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

for f in df_pre.schema:
  print(f)

[結果]
StructField(_c0,IntegerType,true)
StructField(carat,DoubleType,true)
StructField(cut,StringType,true)
StructField(color,StringType,true)
StructField(clarity,StringType,true)
StructField(depth,DoubleType,true)
StructField(table,IntegerType,true)
StructField(price,IntegerType,true)
StructField(x,DoubleType,true)
StructField(y,DoubleType,true)
StructField(z,DoubleType,true)
```

この結果文字列を整形・置換していくと、スキーマのコード化が楽になります。


## Tips 7. 読み込んだデータを俯瞰する

ここまでで、CSVファイルを適切なスキーマを指定して読み込むことができました。










