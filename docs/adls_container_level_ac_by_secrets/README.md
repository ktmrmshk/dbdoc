# Azure Data Lake StorageのContainerレベルでのアクセス制御

Databricksにおいて、
Azure Data Lake Storage(blob store, 以下ADLS)のContainerレベルでのアクセス制御を
実装する方法について説明します。

**注意** ここで説明する方法はAzure DatabricksのPremiumプランが必要になります。Standardプランでは実施できません。

本記事での要件は、以下の通りです。

* 一つのADLSのStorage Account内に複数のContainerが存在する
* それぞれのContainerにアクセスできるグループを限定したい
* ユーザーはいくつかのグループに所属している
* 所属するグループするユーザーは、グループが許可されているADLS Containerにアクセスできる
* それ以外のADLS Containerにはアクセスは許可しない
* ユーザー追加・削除、そのユーザーのグループへの参加・脱退がある程度頻繁に発生する

下図にある構成により、上記の要件を満たすための連携方式を設定していきます。


<img src="resources/ac_overview.png" width="900">

設定のステップは以下の通りです。

0. ADLS StorageおよびContainerを作成する(既存のものを使用する場合、本ステップは不要) 
1. Service Principalを作成する
2. ADLS ContainerのIAMで、"ストレージ BLOB データ共同作成者"ロールに上記のService Principalを追加する
3. Service Principalの"クライアントシークレット"を新規発行する
5. Databricksのユーザーグループを作成する
4. DatabricksのSecrets機能で、新規にSecret Scopeを作成・設定する
    1. 上記のService Principalの"クライアントシークレット"の値を作成したSecret Scopeに登録する
    2. 上記のユーザーグループからこのSecret Scopeへのアクセスを許可する
5. ユーザーグループに所属するユーザーを設定(追加・削除)
6. 各ユーザーでSparkパラメータの設定を実行する
    * 方法1: 毎回、Notebook上で実行する
    * 方法2: クラスタ起動時に設定する


上記は初期設定時のステップになります。ユーザーの追加・削除の際は、最後のステップの操作のみで可能です。


以下、それぞれのステップを見ていきます。

**注意** ここで説明する方法は、Service Principalに割り当てるRoleの範囲をStorage Containerレベルで使用することで、Containerレベルのアクセス制御の分離をしています。このRoleの範囲をStorage　Accountレベルにすることで、同様の方法でStorage Accountごとのアクセス制御が可能になります。 


## Service Principalを作成する

Azure Portalから、
1. => Azure Active Directory
1. => メニュー上部の"+追加"
1. => "アプリの登録"
1. => "名前"にService Principlaの名前を入力(適宜): この例では`dbjp-sp-groupA`を使用する
1. => "登録"をクリック

<img src="resources/sp1.png" width="400">



以上で、Service Pricipalが作成できました。




## ADLS ContainerのIAMで、"ストレージ BLOB データ共同作成者"ロールに上記のService Principal追加する

Azure Portalから

1. => Storage Accountで対象のストレージアカウントの管理画面に入る
1. => 対象のコンテナーを選択し、管理画面に移動する
1. => コンテナー管理画面(ストレージアカウント管理画面ではない!)の左メニュ"アクセス制御(IAM)"をクリック
1. => 上部メニューの"+追加"から、"ロールの割り当て追加"
    * <img src="resources/container1.png" width="400">
1. => 以下の通り選択・設定する
    * ロール: "ストレージ BLOB データ共同作成者"
    * メンバー("+メンバーを選択する"から): 前ステップで作成したService Principal(この例では`
dbjp-sp-groupA`)を選択
    * 最後まで進み、"レビューと作成"

以上で、コンテナーとService Principalの関連づけが完了しました。


## Service Principalの"クライアントシークレット"を新規発行する

Azure Portalから

1. => Azure Active Directory
1. => 左メニュの"アプリの登録"
1. => アプリケーションリストから前ステップで作成したService Principal(`dbjp-sp-groupA`)の管理画面に入る
1. => 左メニューの"証明書とシークレット"
1. => "+新しいクライアントシークレット"
1. => "説明"と"有効期限"を適宜設定し、"追加"
    * <img src="resources/sp_secret.png" width="900">
1. => シークレットの"値"をメモしておく(後ほどDatabricksのSecretsにこの値を登録する)
    * <img src="resources/sp_secret2.png" width="900">


以上でクライアントシークレットの準備ができました。


## Databricksのユーザーグループを作成する

1. => Azure DatabricksにAdminユーザーでログインする
1. => 左メニューの"Settings" > "Admin Console"
1. => Admin Consle上部の"Groups"タブ
1. => "+Create Group"からグループを作成する(ここでは例として`team-A`というグループを作成する)

 <img src="resources/groupa.png" width="600">

以上でユーザーグループが作成できました。


## DatabricksのSecrets機能で、新規にSecret Scopeを作成・設定する

DatabricksのSecrets機能を設定するには、Databricks CLIおよびAccess Tokenの設定が必要になります。SecretsおよびDatabricks CLIの設定方法は以下の記事を参照ください。

[Databricks CLIおよびSecretsの使い方](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/blog/databricks_cli_and_secrets.html)

以降では既にCLIが使用できる前提で進めます。

### 上記のService Principalの"クライアントシークレット"の値を作成したSecret Scopeに登録する

以下の通りのSecretsを作成・設定していきます。

* Secret Scope: `secrets_scope_teamA`
    - Secret #1: 
        + key => `adls_sp_secret`
        + valu e=> "Service Principalのシークレット値" (この例では`EJd7Q~mQE8X3WO6A4tbjGNzBYZWqMoHd-PY3c`)

CLIを実行できるTerminal上で以下のコマンドを実行します。

```bash
### Secrets Scopeの作成
$ databricks secrets create-scope --scope "secrets_scope_teamA"

### Service Principalのシークレット値を登録する
$ databricks secrets put --scope "secrets_scope_teamA" \
    --key "adls_sp_secret" \
    --string-value "EJd7Q~mQE8X3WO6A4tbjGNzBYZWqMoHd-PY3c"

### Scopeの確認
$ databricks secrets list-scopes

Scope                Backend         KeyVault URL
-------------------  --------------  -----------------------------------
secrets_scope_teamA  DATABRICKS      N/A


### Secrets登録の確認
$ databricks secrets list --scope "secrets_scope_teamA"

Key name          Last updated
--------------  --------------
adls_sp_secret   1636267297403
```

以上でService Principalのシークレット値をDatabricks Secretsに登録できました。


### 上記のユーザーグループからこのSecret Scopeへのアクセスを許可する

同様にCLIが実行できるTerminal上で以下のコマンドを実行します。

```bash
### グループ"team-A"に対して`READ`権限を付与する
$ databricks secrets put-acl --scope "secrets_scope_teamA" --principal team-A --permission READ

### 確認
$ databricks secrets list-acls --scope "secrets_scope_teamA" 

Principal                         Permission
--------------------------------  ------------
team-A                            READ
admin-user-foobar@example123.com  MANAGE
```

以上でSecretsがグループ内のユーザーから参照することが可能になりました。


## ユーザーグループに所属するユーザーを設定(追加・削除)

1. => Azure DatabricksにAdminユーザーでログインする
1. => Azure DatabricksにAdminユーザーでログインする
1. => 左メニューの"Settings" > "Admin Console"
1. => Admin Consle上部の"Groups"タブから対象のグループ管理画面に入る(この例では`team-A`)
1. => "Members"タブ内の"+ Add users, groups, or service principals"からグループにユーザーを追加する(または削除する)

以上で、ユーザーのグループへの追加・削除は完了しました。


## 各ユーザーでSparkパラメータの設定を実行する

### 方法1: 毎回、Notebook上で実行する

Notebook上で以下のコードを実行します。
設定するパラメータ変数は以下の通りです。

* `storage_account`: ADLSのStorage Account名
* `storage_countainer`: ADLSのコンテナ名
* `application_id`: Service PrincipalのApplication ID(下記参照)
* `directory_id`: Service PrincipalのDirectory ID(下記参照)
    - Application ID, Directory IDはService Principalの管理画面から確認できます。
    - <img src="resources/apps_id.jpg" width="800">
* `secrets_scope`: Databricks SecretsのScope名
* `secret_key`: Databricks Secretsのシークレットkey名

これらのパラメータ値について、今回の例では以下の通りになります。

```python
[Databricks Notebook上]
%python

# パラメータ設定(適宜書き換えてください)
storage_account = 'dbjpstorageaccount'
storage_container = 'container00aaa'

application_id = '39a6c323-40d4-4e48-9b36-dbb27fedf3ef'
directory_id = '9f37a392-f0ae-4280-9796-f1864a10effc'

secrets_scope = 'secrets_scope_teamA'
secret_key = 'adls_sp_secret'


# 以下はそのまま実行(書き換え不要)
spark.conf.set(f'fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net', 'OAuth')
spark.conf.set(f'fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net', 'org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider')
spark.conf.set(f'fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net', application_id)
spark.conf.set(f'fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net', dbutils.secrets.get(scope=secrets_scope, key=secret_key))
spark.conf.set(f'fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net', f'https://login.microsoftonline.com/{directory_id}/oauth2/token')
```

上記の設定の元、ADLSのコンテナーには以下のパスでアクセスできます。

```
abfss://{storage_container}@{storage_account}.dfs.core.windows.net/
```

アクセスできるか確認します。
```
[Databricks Notebook上]
%python

storage_account = 'dbjpstorageaccount'
storage_container = 'container00aaa'

display(
  dbutils.fs.ls(f'abfss://{storage_container}@{storage_account}.dfs.core.windows.net/')
)
```

<img src="resources/ls_ret.png" width="800">


### 方法2: クラスタ起動時に設定する

Databricksのクラスタ設定で以下のSparkパラメータを設定しておきます。
(`<storage-account-name>`, `<application-id>`, `<directory-id>`を適宜書き換えてください)

SparkパラメータのUI上からSecretsを参照するには`{{secrets/<secret_scope>/<secrets_key>}}`の形式で可能です。今回の例では`{{secrets/secrets_scope_teamA/adls_sp_secret}}`になります。

**注意**  クラスタを起動するユーザーがSecrets Scopeにアクセスできる必要があります。

```
spark.hadoop.fs.azure.account.auth.type.<storage-account-name>.dfs.core.windows.net OAuth
spark.hadoop.fs.azure.account.oauth.provider.type.<storage-account-name>.dfs.core.windows.net org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider
spark.hadoop.fs.azure.account.oauth2.client.id.<storage-account-name>.dfs.core.windows.net <application-id>
spark.hadoop.fs.azure.account.oauth2.client.secret.<storage-account-name>.dfs.core.windows.net {{secrets/secrets_scope_teamA/adls_sp_secret}}
spark.hadoop.fs.azure.account.oauth2.client.endpoint.<storage-account-name>.dfs.core.windows.net https://login.microsoftonline.com/<directory-id>/oauth2/token
```

<img src="resources/spark_conf.png" width="800"> 



先ほどと同様にアクセスできるか確認します。

<img src="resources/ls_ret.png" width="800"> 





## 参考リンク


* [Azure サービス プリンシパルで OAuth 2.0 を使用して Azure Data Lake Storage Gen2 にアクセスする](https://docs.microsoft.com/ja-jp/azure/databricks/data/data-sources/azure/adls-gen2/azure-datalake-gen2-sp-access)
* [Databricks CLIおよびSecretsの使い方](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/blog/databricks_cli_and_secrets.html)
* [Securing access to Azure Data Lake Gen 2 from Azure Databricks](https://github.com/hurtn/datalake-ADLS-access-patterns-with-Databricks/blob/master/readme.md) - 英語
