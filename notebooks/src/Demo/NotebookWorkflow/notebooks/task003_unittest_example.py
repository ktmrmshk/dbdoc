# Databricks notebook source
# MAGIC %md
# MAGIC ### Task003_unittest_example のnotebook
# MAGIC 
# MAGIC * workflowをまとめるnotebook(`master_notebook`)から呼び出される想定
# MAGIC * 処理を実行して、最後に`Job Done by Task001`という文字列を返す

# COMMAND ----------

# 適当な処理
print('Hello! by Task001')

# COMMAND ----------

# DBTITLE 1,単純なAssertの場合
# 単純なAssert文

assert( 1 == 1 )
#assert( 1 == 2 )

# COMMAND ----------

import unittest, io

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    
    def failed_test(self):
        self.assertEqual( 1, 2)

suite = unittest.TestSuite()
suite.addTest(TestStringMethods('test_upper'))
suite.addTest(TestStringMethods('test_isupper'))
suite.addTest(TestStringMethods('test_split'))
suite.addTest(TestStringMethods('failed_test'))

test_results=None
with io.StringIO() as buf:
  runner = unittest.TextTestRunner(stream=buf, verbosity=2)
  runner.run(suite)
  test_results=buf.getvalue()
  
  # for debug
  print(test_results)
  


# COMMAND ----------

# 戻り値を設定できる
dbutils.notebook.exit(test_results)

# COMMAND ----------


