"""Structures for a vareity of different test results."""


class Result(object):
  """Information about the results of the test.

  Args:
      test_id (str): Id when combined with test_source should represent a unique
      test that maybe run on multiple system types, e.g. P100 or K80.
      test_name (str): Simplified test name that does not have to be unique and
        may
      be useful for display purposes.
      result (float): Float value representing the result of the test.
      result_type (str): Type of result, total_time, exps_per_sec,
        oom_batch_size, or
      global_step_per_sec.
      test_source (str, optional): Test collection, e.g. tf_cnn_benchmarks,
        keras_benchmarks,
      model_garden_convergence, or caffe2_bench.
      result_sounce (str, optional): Source of the test, e.g. DGX-cluster,
        keras_test_cluster,
      or manual

  """

  def __init__(self,
               test_id,
               test_name,
               result,
               result_type='total_time',
               test_source='tf_cnn_bench',
               result_source='unknown'):
    self.test_id = unicode(test_id)
    self.test_name = unicode(test_name)
    self.result = result
    self.result_type = unicode(result_type)
    self.test_source = unicode(test_source)
    self.result_source = unicode(result_source)


class SystemInfo(object):
  """Information about the system the test was executed on.

  Args:
      platform (str): Higher level platform, e.g. aws, gce, or workstation.
      platform_type (str): Type of platform, DGX-1, p3.8xlarge, or z420.
      accel_type (str, optional): Type of accelerator, e.g. K80 or P100.
      cpu_cores (str, optional): Number of physical cpu cores.
      cpu_type (str, optional): Type of cpu.

  """

  def __init__(self,
               platform=None,
               platform_type=None,
               accel_type=None,
               cpu_cores=None,
               cpu_type=None):

    if platform:
      self.platform = unicode(platform)
    if platform_type:
      self.platform_type = unicode(platform_type)
    if accel_type:
      self.accel_type = unicode(accel_type)
    if cpu_cores:
      self.cpu_cores = cpu_cores
    if cpu_type:
      self.cpu_type = unicode(cpu_type)


class TestInfo(object):

  def __init__(self,
               framework='tensorflow',
               batch_size=None,
               model=None,
               accel_cnt=None):
    """Initialize TestInfo object.

    Args:
      framework (str, optional): Framework being tested, e.g. tesnsorflow,
      mxnet, or caffe2.  Defaults to tensorflow.
      batch_size (int, optional): Total batch size.
      model: Model being tested.
      accel_cnt (int, optional): Number of accelerators bieng utilized.
    """
    if framework:
      self.framework = framework
    if batch_size:
      self.batch_size = batch_size
    if model:
      self.model = model
    if accel_cnt:
      self.accel_cnt = accel_cnt
