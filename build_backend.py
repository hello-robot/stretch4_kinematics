import os
import setuptools.build_meta as _orig

def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    os.environ["IS_EDITABLE_INSTALL"] = "1"
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)

def get_requires_for_build_editable(config_settings=None):
    os.environ["IS_EDITABLE_INSTALL"] = "1"
    return _orig.get_requires_for_build_editable(config_settings)

def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    os.environ["IS_EDITABLE_INSTALL"] = "1"
    return _orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)

# Delegate all other build hooks (build_sdist, build_wheel, etc.) to setuptools.build_meta
def __getattr__(name):
    return getattr(_orig, name)
