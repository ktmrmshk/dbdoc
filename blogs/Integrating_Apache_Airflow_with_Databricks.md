# Apache AirflowとDatabricksの連携 - AirflowでDatabricksのワークロードを管理するための簡単なステップ・バイ・ステップのチュートリアル

投稿 Andrew Chen 投稿：エンジニアリングのブログJuly 19, 2017

*このブログ記事は、Databricksプラットフォーム、インフラストラクチャ管理、統合、ツール、監視、プロビジョニングに関する社内エンジニアリングブログのシリーズの一部です。*

------

本日、オープンソースのワークフロースケジューラーとして人気のある[Apache Airflow](https://airflow.incubator.apache.org/)にDatabricksがネイティブに統合されたことを発表しました。このブログ記事では、Airflowをセットアップし、Databricksのジョブをトリガーするために使用する方法を説明します。


[Databricks社のUnified Data Analytics Platform（UAP）](https://databricks.com/jp/product/unified-data-analytics-platform)の非常に人気のある機能の1つに、データサイエンスノートブックを定期実行できる本番ジョブに直接変換できる機能があります(Job)。この機能により、探索的データサイエンスから本番データエンジニアリングまでのワークフローが統一されますが、データエンジニアリングジョブの中には、ノートブックでは捉えにくい複雑な依存関係が含まれている場合があります。このような複雑なユースケースをサポートするために、REST APIを提供し、ノートブックやライブラリに基づくジョブが外部システムからトリガーされるようにしています。その中でも、お客様に最もよく使われているスケジューラの一つがAirflowです。今回、Airflowを拡張し、Databricksをすぐにサポートできるようになりました。


## Airflowの基礎

Airflowは、依存関係管理機能を備えた汎用のワークフロースケジューラです。定期的なジョブのスケジューリングに加え、データパイプラインの異なるステージ間の明示的な依存関係を表現することができます。


各[ETLパイプライン](https://databricks.com/jp/glossary/etl-pipeline)は、タスクの有向非環状グラフ（DAG）として表現されます（SparkのDAGスケジューラやタスクとの違いはありません）。依存関係はエッジによってDAGにエンコードされます。どのエッジでも、上流のタスクが正常に完了した場合のみ、下流のタスクがスケジュールされます。例えば、以下のDAGの例では、タスクAが正常に完了した後に、タスクBとCが起動されます。タスクDは、タスクBとCの両方が正常に完了したときに起動されます。

![dag](https://databricks.com/wp-content/uploads/2017/07/image1-2.png)

Airflow のタスクは、`operator`クラスのインスタンスで、小さなPythonスクリプトとして実装されています。例えば、成功する前に前提条件が真であるかどうかをポーリングしたり（センサーとも呼ばれる）、ETLを直接実行したり、Databricksのような外部システムをトリガーしたりすることができます。

> Airflowについて、詳しくは公式の[ドキュメント](http://airflow.readthedocs.io/en/latest/)を参照ください。


## AirflowでのDatabricksのネイティブなインテグレーション


[`DatabricksSubmitRunOperator`](https://airflow.apache.org/docs/apache-airflow-providers-databricks/stable/operators.html)というAirflowのオペレーターを実装し、AirflowとDatabricksのよりスムーズな統合を可能にしました。このオペレーターを介して、Databricksの[Runs Submit](https://docs.databricks.com/dev-tools/api/latest/jobs.html#jobsjobsservicesubmitrun) APIエンドポイントを叩くことができ、外部からjar、pythonスクリプト、またはノートブックの単一の実行をトリガーすることができます。ランを送信する最初のリクエストを行った後、オペレーターはランの結果をポーリングし続けます。実行が正常に完了すると、オペレータは下流のタスクの実行を許可して戻ります。

DatabricksSubmitRunOperatorをオープンソースのAirflowプロジェクトにアップストリームでコントリビュートしました。しかし、統合はAirflow 1.9.0がリリースされるまで、リリースブランチに切り込まれることはありません。それまでは、このオペレータを使用するには、Databricks のフォークである Airflow をインストールすることができます。これは基本的に Airflow バージョン 1.8.1 に私たちの DatabricksSubmitRunOperator パッチを適用したものです。

```bash
pip install --upgrade "git+git://github.com/databricks/incubator-airflow.git@1.8.1-db1#egg=apache-airflow[databricks]"
```

## AirflowのDatabricks連携のチュートリアル


このチュートリアルでは、ローカルマシン上で動作するおもちゃのAirflow 1.8.1のデプロイメントを設定し、Databricksで動作するトリガーとなるDAGの例をデプロイします。


まず最初に行うことは、sqlite データベースの初期化です。Airflow は、雑多なメタデータを追跡するためにこのデータベースを使用します。Airflow の本番環境では、設定を編集して Airflow に MySQL や Postgres のデータベースを指定することになりますが、今回のおもちゃの例では、単純にデフォルトの sqlite データベースを使用します。初期化を実行するには、次のようにします。


```bash
airflow initdb
```

Airflow のデプロイメントのための SQLite データベースとデフォルト設定は、`~/airflow` で初期化されます。


次のステップでは、1つの線形依存関係を持つ2つのDatabricksジョブを実行するDAGを書きます。1つ目のDatabricksジョブは、`/Users/airflow@example.com/PrepareData`にあるノートブックを起動し、2つ目のジョブは、`dbfs:/lib/etl-0.1.jar`にあるjarを実行します。

マイルハイから見ると、スクリプトDAGは基本的に2つの`DatabricksSubmitRunOperator`タスクを構築し、最後に`set_dowstream`メソッドで依存関係を設定しています。コードのスケルトンバージョンは次のようなものです。

```python
notebook_task = DatabricksSubmitRunOperator(
    task_id='notebook_task',
    ...)

spark_jar_task = DatabricksSubmitRunOperator(
    task_id='spark_jar_task',
    ...)
notebook_task.set_downstream(spark_jar_task)
```

実際には、DAGファイルを動作させるためには、他にもいくつかの詳細を記入する必要があります。最初のステップは、DAG内の各タスクに適用されるデフォルトの引数を設定することです。

```python
args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2)
}
```

ここで興味深いのは`depends_on_past`と`start_date`の2つの引数です。`depends_on_past`が`true`の場合、タスクの前のインスタンスが正常に完了しない限り、タスクを起動してはいけないというシグナルをAirflowに送ります。`start_date`引数は、最初のタスクインスタンスがいつスケジュールされるかを決定します。



DAGスクリプトの次のセクションでは、実際にDAGをインスタンス化します。

```python
dag = DAG(
    dag_id='example_databricks_operator', default_args=args,
    schedule_interval='@daily')
```

このDAGでは、ユニークなIDを与え、先に宣言したデフォルトの引数を付け、毎日のスケジュールを与えます。次に、タスクを実行するクラスタの仕様を指定します。

```python
new_cluster = {
    'spark_version': '2.1.0-db3-scala2.11',
    'node_type_id': 'r3.xlarge',
    'aws_attributes': {
        'availability': 'ON_DEMAND'
    },
    'num_workers': 8
}
```

この仕様のスキーマは、Runs Submitエンドポイントの新しいクラスタフィールドと一致しています。DAGの例では、ワーカーの数を減らしたり、インスタンスサイズをより小さいものに変更することができます。

最後に、`DatabricksSubmitRunOperator`をインスタンス化して、DAGに登録します。

```python
notebook_task_params = {
    'new_cluster': new_cluster,
    'notebook_task': {
        'notebook_path': '/Users/airflow@example.com/PrepareData',
    },
}
```

```python
# Example of using the JSON parameter to initialize the operator.
notebook_task = DatabricksSubmitRunOperator(
    task_id='notebook_task',
    dag=dag,
    json=notebook_task_params)
```

このコードでは、JSONパラメータは、Runs SubmitエンドポイントにマッチするPythonのdictionary(`dict型`)を取ります。

このタスクの下流に別のタスクを追加するには、`DatabricksSubmitRunOperator`を再度インスタンス化し、`notebook_task operator`インスタンスの特別な`set_downstream`メソッドを使用して依存関係を登録します。

```python
# Example of using the named parameters of DatabricksSubmitRunOperator
# to initialize the operator.
spark_jar_task = DatabricksSubmitRunOperator(
    task_id='spark_jar_task',
    dag=dag,
    new_cluster=new_cluster,
    spark_jar_task={
        'main_class_name': 'com.example.ProcessData'
    },
    libraries=[
        {
            'jar': 'dbfs:/lib/etl-0.1.jar'
        }
    ]
)

notebook_task.set_downstream(spark_jar_task)
```


このタスクは、`dbfs:/lib/etl-0.1.jar`にあるjarを実行します。

notebook_taskでは、JSONパラメータを使用してsubmit runエンドポイントの完全な仕様を指定し、spark_jar_taskではsubmit runエンドポイントのトップレベルのキーを`DatabricksSubmitRunOperator`のパラメータにフラット化していることに注目してください。オペレータをインスタンス化する両方の方法は同等ですが、後者の方法では、[`spark_python_task`](https://docs.databricks.com/dev-tools/api/latest/jobs.html#request-structure)や`spark_submit_task`のような新しいトップレベルのフィールドを使用することはできません。`DatabricksSubmitRunOperator`の完全なAPIについての詳細な情報は、[こちらのドキュメント](http://apache-airflow-docs.s3-website.eu-central-1.amazonaws.com/docs/apache-airflow-providers-databricks/latest/operators.html)をご覧ください。

DAGができたので、Airflowにインストールするために、`~/airflow`に`~/airflow/dags`というディレクトリを作成し、そのディレクトリにDAGをコピーします。

この時点で、AirflowはDAGをピックアップすることができるはずです。


```bash
$ airflow list_dags                                                           [10:27:13]
[2017-07-06 10:27:23,868] {__init__.py:57} INFO - Using executor SequentialExecutor
[2017-07-06 10:27:24,238] {models.py:168} INFO - Filling up the DagBag from /Users/andrew/airflow/dags


-------------------------------------------------------------------
DAGS
-------------------------------------------------------------------
example_bash_operator
example_branch_dop_operator_v3
example_branch_operator
example_databricks_operator
```

また、Web UIでDAGを可視化することもできます。起動するには、`airflow webserver`を実行し、`localhost:8080`に接続します。`example_databricks_operator`をクリックすると、DAGのビジュアライズがたくさん表示されます。ここではその例を紹介します。

![dagpreview](https://databricks.com/wp-content/uploads/2017/07/image2-2.png)


この時点で、注意深い人は、DAG内のどこにもDatabricksシャードへのホスト名、ユーザー名、パスワードなどの情報が指定されていないことにも気づくでしょう。これを設定するには、データベースに保存されている資格情報をDAGから参照することができるAirflowの[connect](https://airflow.incubator.apache.org/concepts.html?highlight=connections#connections)プリミティブを使用します。デフォルトでは、すべての`DatabricksSubmitRunOperator`は、[`databricks_conn_id`](http://apache-airflow-docs.s3-website.eu-central-1.amazonaws.com/docs/apache-airflow/latest/integration.html)パラメータを `databricks_default`に設定しているので、今回のDAGでは、`ID`が`databricks_default`のコネクションを追加する必要があります。


これを行う最も簡単な方法は、ウェブUIからです。上部の「Admin」をクリックして、ドロップダウンの「Connections」をクリックすると、現在のすべての接続が表示されます。今回の使用例では、"databricks_default "に接続を追加します。最終的な接続は以下のようになります。


![ui](https://databricks.com/wp-content/uploads/2017/07/image3-2.png)

DAG の設定がすべて完了したところで、各タスクをテストしてみましょう。`notebook_task`については、`airflow test example_databricks_operator notebook_task 2017-07-01`、`spark_jar_task`については、`airflow test example_databricks_operator spark_jar_task 2017-07-01`を実行します。DAGをスケジュール通りに実行するには、`airflow scheduler`というコマンドでスケジューラデーモンプロセスを起動します。

すべてがうまくいくと、スケジューラーを起動した後、Web UIでDAGのバックフィル実行が始まるのを確認できます。

## 次のステップ

結論として、このブログ記事はAirflowとDatabricksの統合をセットアップする簡単な例を提供しています。AirflowへのDatabricksの拡張と統合により、Databricksの[Runs Submit](https://docs.databricks.com/dev-tools/api/latest/jobs.html#runs-submit) APIを介したアクセスが可能となり、Databricksプラットフォーム上で計算を呼び出す方法を示しています。本番環境でのAirflow導入の設定方法については、[Airflowの公式ドキュメント](https://airflow.apache.org/docs/stable/)をご覧ください。

