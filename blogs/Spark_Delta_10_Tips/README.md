# Sparkを使ったデータ分析・処理の書き方 - 10のTips

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

おや、結果をみると何かおかしいですね。一行目がヘッダの内容になっているのがわかります。実際にCSVファイルを見てみると以下のようになっています。

```csv
"","carat","cut","color","clarity","depth","table","price","x","y","z"
"1",0.23,"Ideal","E","SI2",61.5,55,326,3.95,3.98,2.43
"2",0.21,"Premium","E","SI1",59.8,61,326,3.89,3.84,2.31
"3",0.23,"Good","E","VS1",56.9,65,327,4.05,4.07,2.31
"4",0.29,"Premium","I","VS2",62.4,58,334,4.2,4.23,2.63
...
```

Sparkで読み込むときに"CSVの一行目はHeader情報"というオプション`header=True`を指定してみます。

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

好みの問題ではあると思いますが、個人的には上記の書き方の方が、

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
それでは読み込んだデータを俯瞰してみましょう。`summary()`でデータの統計を見てます。

```
df.summary().show()

[結果]
+-------+------------------+------------------+---------+-----+-------+------------------+------------------+-----------------+------------------+------------------+------------------+
|summary|               _c0|             carat|      cut|color|clarity|             depth|             table|            price|                 x|                 y|                 z|
+-------+------------------+------------------+---------+-----+-------+------------------+------------------+-----------------+------------------+------------------+------------------+
|  count|             53940|             53940|    53940|53940|  53940|             53940|             53016|            53940|             53940|             53940|             53940|
|   mean|           26970.5|0.7979397478679852|     null| null|   null| 61.74940489432624|57.476063829787236|3932.799721913237| 5.731157211716609| 5.734525954764462|3.5387337782723316|
| stddev|15571.281096942537|0.4740112444054196|     null| null|   null|1.4326213188336525| 2.225280761946626|3989.439738146397|1.1217607467924915|1.1421346741235616|0.7056988469499883|
|    min|                 1|               0.2|     Fair|    D|     I1|              43.0|                43|              326|               0.0|               0.0|               0.0|
|    25%|             13482|               0.4|     null| null|   null|              61.0|                56|              950|              4.71|              4.72|              2.91|
|    50%|             26966|               0.7|     null| null|   null|              61.8|                57|             2401|               5.7|              5.71|              3.53|
|    75%|             40451|              1.04|     null| null|   null|              62.5|                59|             5324|              6.54|              6.54|              4.04|
|    max|             53940|              5.01|Very Good|    J|   VVS2|              79.0|                95|            18823|             10.74|              58.9|              31.8|
+-------+------------------+------------------+---------+-----+-------+------------------+------------------+-----------------+------------------+------------------+------------------+
```

`mean`と`stddev`の表示で表全体が見えづらくなる場合は、それらを除いくと見やすくなります。

```
df.summary("count", "min", "25%", "50%", "75%", "max").show()

[結果]
+-------+-----+-----+---------+-----+-------+-----+-----+-----+-----+-----+-----+
|summary|  _c0|carat|      cut|color|clarity|depth|table|price|    x|    y|    z|
+-------+-----+-----+---------+-----+-------+-----+-----+-----+-----+-----+-----+
|  count|53940|53940|    53940|53940|  53940|53940|53016|53940|53940|53940|53940|
|    min|    1|  0.2|     Fair|    D|     I1| 43.0|   43|  326|  0.0|  0.0|  0.0|
|    25%|13482|  0.4|     null| null|   null| 61.0|   56|  950| 4.71| 4.72| 2.91|
|    50%|26966|  0.7|     null| null|   null| 61.8|   57| 2401|  5.7| 5.71| 3.53|
|    75%|40451| 1.04|     null| null|   null| 62.5|   59| 5324| 6.54| 6.54| 4.04|
|    max|53940| 5.01|Very Good|    J|   VVS2| 79.0|   95|18823|10.74| 58.9| 31.8|
+-------+-----+-----+---------+-----+-------+-----+-----+-----+-----+-----+-----+
```

この結果からいろいろ読み取れます。例えば、欠損値(null値)の観点から、`table`カラムだけnull値が含まれる、などがわかります。

実際に`table`カラムが`null`になっているレコード数を確認していきましょう。

```
df.where('table is null').count()

[結果]
924
```

`53940 - 53016 = 924`ですから、整合性がとれています。


## Tips 8. データフレーム内の特定のフィールドの値を取り出す

データフレームの処理はデータ全体を一括処理(Spark!)できる一方で、データフレーム内の特定のフィールドの値を取り出して、Pythonコードの中で使いたい場合がしばしばあります。
例えば、上記で使用した`df.summary()`は、それ自体データフレームとして統計情報をまとめており、この中で`table`カラムの`mean`と`stddev`の値をコードの中で参照したい場合などです。

実際に取り出してみましょう。
collect()を使ってデータを集めて、その後、値を拾います。

```
# カラムの`table`
# `df.summary()`で`mean`, `stddev`, `50%`はそれぞれRow 1, Row 2, Row 5
table_mean   = df.summary().collect()[1]['table']
table_stddev = df.summary().collect()[2]['table']
table_median = df.summary().collect()[5]['table']

print(f'mean   => {table_mean}')
print(f'stddev => {table_stddev}')
print(f'median => {table_median}')

[結果]
mean   => 57.476063829787236
stddev => 2.225280761946626
median => 57
```

値が取り出せていることが確認できました。


## Tips 9. 欠損値の対応をする


欠損値の対応はいくつかのパターンがあります。ここでは、null値が含まれる行(レコード)を削除する方法と、中央値で埋める方法を見ていきたいと思います。

### nullを含むレコードを削除

まずは、null値のレコードを削除する方法は`dropna()`関数で可能です。簡単です。

  ```
df_dropped = df.dropna()
df_dropped.summary('count').show()

[結果]
+-------+-----+-----+-----+-----+-------+-----+-----+-----+-----+-----+-----+
|summary|  _c0|carat|  cut|color|clarity|depth|table|price|    x|    y|    z|
+-------+-----+-----+-----+-----+-------+-----+-----+-----+-----+-----+-----+
|  count|53016|53016|53016|53016|  53016|53016|53016|53016|53016|53016|53016|
+-------+-----+-----+-----+-----+-------+-----+-----+-----+-----+-----+-----+
```

全てのカラム(not null)数が`53016`になりました。もちろん、`null`な`table`カラムのレコード数はゼロです。

```
df_dropped.where('table is null').count()

[結果]
0
```

### 中央値で埋める

続いて、中央値で埋める方法です。
先ほど取り出した`table_median`をそのまま使います。

```
df_imputed = df.na.fill(  int(table_median), 'table')
df_imputed.summary('count').show()

[結果]
+-------+-----+-----+-----+-----+-------+-----+-----+-----+-----+-----+-----+
|summary|  _c0|carat|  cut|color|clarity|depth|table|price|    x|    y|    z|
+-------+-----+-----+-----+-----+-------+-----+-----+-----+-----+-----+-----+
|  count|53940|53940|53940|53940|  53940|53940|53940|53940|53940|53940|53940|
+-------+-----+-----+-----+-----+-------+-----+-----+-----+-----+-----+-----+
```

レコード数だけでみると、値が埋められたように見えます。
実際にnullフィールドが中央値になっているか確認してみましょう。

```
# (Before) nullのフィールド
df.where('table is null').limit(3).show()

[結果]
+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+
|_c0|carat|      cut|color|clarity|depth|table|price|   x|   y|   z|
+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+
| 67| 0.32|    Ideal|    I|   VVS1| 62.0| null|  553|4.39|4.42|2.73|
|178| 0.72|Very Good|    G|    VS2| 63.7| null| 2776|5.62|5.69|3.61|
|185| 0.72|     Good|    G|    VS2| 59.7| null| 2776| 5.8|5.84|3.47|
+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+


# (After) 埋められたフィールド
df_imputed.where('_c0 in (67, 178, 185)').show()

[結果]
+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+
|_c0|carat|      cut|color|clarity|depth|table|price|   x|   y|   z|
+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+
| 67| 0.32|    Ideal|    I|   VVS1| 62.0|   57|  553|4.39|4.42|2.73|
|178| 0.72|Very Good|    G|    VS2| 63.7|   57| 2776|5.62|5.69|3.61|
|185| 0.72|     Good|    G|    VS2| 59.7|   57| 2776| 5.8|5.84|3.47|
+---+-----+---------+-----+-------+-----+-----+-----+----+----+----+
```

確認できました。



## Tips 10. PandasとSparkを交互に行き来する

ここまではSparkのデータフレームの操作方法を見てきました。Sparkのデータフレームは処理のバックエンドがSpark分散処理によってペタバイト級のデータフレームでも扱えるスケーラビリティがある一方で、Pandasのデータフレームを受け付けるツール群を使い場合がよくあります。そうした場合には、SparkのデータフレームとPandasのデータフレームは関数一つで相互に変換可能です。

```
# Spark Dataframe -> Pandas Dataframe
pandas_df = df.toPandas()
print( pandas_df.head() )

[結果]
   _c0  carat      cut color clarity  depth  table  price     x     y     z
0    1   0.23    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43
1    2   0.21  Premium     E     SI1   59.8   61.0    326  3.89  3.84  2.31
2    3   0.23     Good     E     VS1   56.9   65.0    327  4.05  4.07  2.31
3    4   0.29  Premium     I     VS2   62.4   58.0    334  4.20  4.23  2.63
4    5   0.31     Good     J     SI2   63.3   58.0    335  4.34  4.35  2.75
```

```
# Pandas Dataframe -> Spark Dataframe
spark_df = spark.createDataFrame(pandas_df)
spark_df.show(5)

[結果]
+---+-----+-------+-----+-------+-----+-----+-----+----+----+----+
|_c0|carat|    cut|color|clarity|depth|table|price|   x|   y|   z|
+---+-----+-------+-----+-------+-----+-----+-----+----+----+----+
|  1| 0.23|  Ideal|    E|    SI2| 61.5| 55.0|  326|3.95|3.98|2.43|
|  2| 0.21|Premium|    E|    SI1| 59.8| 61.0|  326|3.89|3.84|2.31|
|  3| 0.23|   Good|    E|    VS1| 56.9| 65.0|  327|4.05|4.07|2.31|
|  4| 0.29|Premium|    I|    VS2| 62.4| 58.0|  334| 4.2|4.23|2.63|
|  5| 0.31|   Good|    J|    SI2| 63.3| 58.0|  335|4.34|4.35|2.75|
+---+-----+-------+-----+-------+-----+-----+-----+----+----+----+
only showing top 5 rows
```

大きいデータ処理はSparkデータフレームで実施し、集約できたデータをPandasデータフレームにして各種ツールに喰わせるなどの連携がスムーズにできます。


## むすび

いかがでしょうか。今回は主にSparkを使ってデータ分析・処理する場合の10のTipsを紹介いたしました。
次回はDelta lakeも使用したSparkでのデータ分析のTipsを紹介いたします。


## 参考

* [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/reference/index.html)
* [Spark Document - CSV Files](https://spark.apache.org/docs/latest/sql-data-sources-csv.html)



