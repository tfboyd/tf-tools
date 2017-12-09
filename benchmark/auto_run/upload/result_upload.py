"""Upload test results."""
import copy
from datetime import datetime
import json
import os

import pytz
import result_info

import google.auth
from google.cloud import bigquery
from google.cloud.bigquery.dbapi import connect


def upload_result(result,
                  project,
                  dataset,
                  table,
                  test_info=None,
                  system_info=None,
                  extras=None,
                  debug_level=0):
  """Upload test result.

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.

  Args:
    result: Result of the test, strongly suggest use Result.
    project: Project where BigQuery table is located
    dataset: BigQuery dataset to use.
    table: BigQuery table to insert into.
    test_info: Additional test info, suggested use or extend TestInfo.
    system_info: Extra system info, suggested use or extend SystemInfo.
    extras: Dictionary of values that will be serialized to JSON.
    debug_level: Set to 1 for debug info.
  """

  # Project is disgarded in favor of what the user passes in.
  credentials, _ = google.auth.default()

  row = copy.copy(result)
  # The user is set to the email address of the service account.  If that is not
  # found, then the logged in user is used as a last best guess.
  if not hasattr(row, 'user'):
    if hasattr(credentials, 'service_account_email'):
      row.user = credentials.service_account_email
    else:
      row.user = unicode(os.getlogin())

  # gpylint warning suggests using a different lib that does not look helpful.
  # pylint: disable=W6421
  setattr(row, 'timestamp', datetime.utcnow().replace(tzinfo=pytz.utc))

  # Convert extra info into JSON.
  system_info_json = json.dumps(vars(system_info) if system_info else None)
  test_info_json = json.dumps(vars(test_info) if test_info else None)
  extras_json = json.dumps(extras)
  # BigQuery expects unicode object and maps that to datatype.STRING.
  setattr(row, 'system_info', unicode(system_info_json))
  setattr(row, 'test_info', unicode(test_info_json))
  setattr(row, 'extras', unicode(extras_json))

  client = bigquery.Client(project=project, credentials=credentials)
  conn = connect(client=client)
  cursor = conn.cursor()
  sql = """INSERT into {}.{} (test_id, test_name,
             test_source, result_source, result, result_type, user,
             timestamp, system_info, test_info, extras)
             VALUES (@test_id, @test_name, @test_source, @result_source,
             @result, @result_type, @user, @timestamp, @system_info, @test_info,
             @extras)""".format(dataset, table)

  cursor.execute(sql, parameters=vars(row))
  conn.commit()
  # Cursor and connection closes on their own as well.
  cursor.close()
  conn.close()
