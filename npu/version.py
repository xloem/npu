import pkg_resources

__version__ = "0.3.900"

from npu.core.common import npu_print

_module = "npu"


latest_version = pkg_resources.get_distribution(_module).version
if latest_version != __version__:
    npu_print("Current version of npu library is {}. Latest version on pypi is {}".format(__version__, latest_version))
