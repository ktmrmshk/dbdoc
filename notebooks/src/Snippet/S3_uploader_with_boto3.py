# Databricks notebook source
#params
params={}
params['geolite2']={}
params['geolite2']['license_key'] = dbutils.secrets.get(scope='poc', key='geolite_license_key')
params['geolite2']['asn_csv_url'] = f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-ASN-CSV&license_key={params['geolite2']['license_key']}&suffix=zip"
params['geolite2']['country_csv_url'] =f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City-CSV&license_key={params['geolite2']['license_key']}&suffix=zip"
params['geolite2']['city_csv_url'] = f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country-CSV&license_key={params['geolite2']['license_key']}&suffix=zip"
params['geolite2']['s3_base_dir'] = 'geolite2'
params['geolite2']['asn_dirname']='ASN'
params['geolite2']['country_dirname']='Country'
params['geolite2']['city_dirname']='City'
params['geolite2']['s3_bucket'] = 'databricks-ktmr-s3'

# COMMAND ----------

def upload_to_s3(local_filepath, bucket, upload_path):
  import boto3
  from botocore.exceptions import ClientError
  client = boto3.client('s3')
  try:
    ret=client.upload_file(local_filepath, bucket, upload_path)
  except ClientError as e:
    print(e)
    return False
  return True


def downlaod_file(url, local_filepath):
  import subprocess
  try:
    ret=subprocess.Popen(args=['curl', '-o', local_filepath, url],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE )
    print( ret.communicate() )
  except Exception as e:
    print(f'error=>{e}')
    

def get_and_upload_csv_to_s3(url, tagname, s3_bucket, upload_dir, tmpdir='/tmp'):
  '''
  url: geolite2 permalink
  tagname: fike name for temp use
  upload_dir: s3 directory name the file is uploded to
  '''
  import zipfile
  import os.path
  
  # get the files
  local_filepath = os.path.join(tmpdir, tagname)
  print(f'local_filepath => {local_filepath}')
  downlaod_file(url, local_filepath)
  
  # unzip
  z = zipfile.ZipFile(local_filepath)
  files = z.namelist()
  
  with z as f:
    f.extractall(tmpdir)
    
  # uploading to s3
  for n in files:
    if n.lower().endswith('.csv'):
      local=os.path.join(tmpdir, n)
      base=os.path.basename(n)
      s3_path=os.path.join(upload_dir, tagname, base)
      print(f'uploading files to s3: {local} => {s3_path}')
      upload_to_s3(local, s3_bucket, s3_path)


# COMMAND ----------

# ASN
get_and_upload_csv_to_s3(
  params['geolite2']['asn_csv_url'],
  params['geolite2']['asn_dirname'],
  params['geolite2']['s3_bucket'],
  params['geolite2']['s3_base_dir']
)

# Country
get_and_upload_csv_to_s3(
  params['geolite2']['country_csv_url'],
  params['geolite2']['country_dirname'],
  params['geolite2']['s3_bucket'],
  params['geolite2']['s3_base_dir']
)


# City
get_and_upload_csv_to_s3(
  params['geolite2']['city_csv_url'],
  params['geolite2']['city_dirname'],
  params['geolite2']['s3_bucket'],
  params['geolite2']['s3_base_dir']
)


