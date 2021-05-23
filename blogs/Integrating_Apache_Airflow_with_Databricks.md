# Apache AirflowとDatabricksの連携 - AirflowでDatabricksのワークロードを管理するための簡単なステップ・バイ・ステップのチュートリアル

投稿 Andrew Chen 投稿：エンジニアリングのブログJuly 19, 2017

*このブログ記事は、Databricksプラットフォーム、インフラストラクチャ管理、統合、ツール、監視、プロビジョニングに関する社内エンジニアリングブログのシリーズの一部です。*

------

本日、オープンソースのワークフロースケジューラーとして人気のある[Apache Airflow](https://airflow.incubator.apache.org/)にDatabricksがネイティブに統合されたことを発表しました。このブログ記事では、Airflowをセットアップし、Databricksのジョブをトリガーするために使用する方法を説明します。


[Databricks社のUnified Data Analytics Platform（UAP）](https://databricks.com/jp/product/unified-data-analytics-platform)の非常に人気のある機能の1つに、データサイエンスノートブックを定期実行できる本番ジョブに直接変換できる機能があります(Job)。この機能により、探索的データサイエンスから本番データエンジニアリングまでのワークフローが統一されますが、データエンジニアリングジョブの中には、ノートブックでは捉えにくい複雑な依存関係が含まれている場合があります。このような複雑なユースケースをサポートするために、REST APIを提供し、ノートブックやライブラリに基づくジョブが外部システムからトリガーされるようにしています。その中でも、お客様に最もよく使われているスケジューラの一つがAirflowです。今回、Airflowを拡張し、Databricksをすぐにサポートできるようになりました。


## Airflowの基礎

Airflowは、依存関係管理機能を備えた汎用のワークフロースケジューラです。定期的なジョブのスケジューリングに加え、データパイプラインの異なるステージ間の明示的な依存関係を表現することができます。


各ETLパイプラインは、タスクの有向非環状グラフ（DAG）として表現されます（SparkのDAGスケジューラやタスクとの違いはありません）。依存関係はエッジによってDAGにエンコードされます。どのエッジでも、上流のタスクが正常に完了した場合のみ、下流のタスクがスケジュールされます。例えば、以下のDAGの例では、タスクAが正常に完了した後に、タスクBとCが起動されます。タスクDは、タスクBとCの両方が正常に完了したときに起動されます。

![dag](https://databricks.com/wp-content/uploads/2017/07/image1-2.png)

Airflow のタスクは、`operator`クラスのインスタンスで、小さなPythonスクリプトとして実装されています。例えば、成功する前に前提条件が真であるかどうかをポーリングしたり（センサーとも呼ばれる）、ETLを直接実行したり、Databricksのような外部システムをトリガーしたりすることができます。

> Airflowについて、詳しくは公式の[ドキュメント](http://airflow.readthedocs.io/en/latest/)を参照ください。


## AirflowでのDatabricksのネイティブなインテグレーション


`DatabricksSubmitRunOperator`というAirflowのオペレーターを実装し、AirflowとDatabricksのよりスムーズな統合を可能にしました。このオペレーターを介して、Databricks Runs Submit APIエンドポイントを叩くことができ、外部からjar、pythonスクリプト、またはノートブックの単一の実行をトリガーすることができます。ランを送信する最初のリクエストを行った後、オペレーターはランの結果をポーリングし続けます。実行が正常に完了すると、オペレータは下流のタスクの実行を許可して戻ります。

DatabricksSubmitRunOperatorをオープンソースのAirflowプロジェクトにアップストリームでコントリビュートしました。しかし、統合はAirflow 1.9.0がリリースされるまで、リリースブランチに切り込まれることはありません。それまでは、このオペレータを使用するには、Databricks のフォークである Airflow をインストールすることができます。これは基本的に Airflow バージョン 1.8.1 に私たちの DatabricksSubmitRunOperator パッチを適用したものです。

```
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
    …)

spark_jar_task = DatabricksSubmitRunOperator(
    task_id='spark_jar_task',
    …)
notebook_task.set_downstream(spark_jar_task)
```