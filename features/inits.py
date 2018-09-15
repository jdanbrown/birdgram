import subprocess

import cloudpickle
import structlog

from logging_ import init_logging  # For export


def init_check_deps():
    """Raise if any deps aren't found"""
    shell = lambda cmd: subprocess.check_call(cmd, shell=True)
    shell('ffmpeg -version >/dev/null 2>&1')  # Is ffmpeg installed?
    shell('ffmpeg -version 2>&1 | grep libmp3lame >/dev/null')  # Was ffmpeg built with libmp3lame?


def init_cloudpickle():
    """Add cloudpickle dispatches"""
    # Fix structlog loggers to be cloudpickle-able [like https://github.com/cloudpipe/cloudpickle/pull/96]
    #   - TODO Open a cloudpickle issue [just requires making a small repro to illustrate the problem]
    def save_structlog_BoundLoggerLazyProxy(self, obj):
        self.save_reduce(structlog.get_logger, obj._logger_factory_args, obj=obj)
    def save_structlog_BoundLoggerBase(self, obj):
        # TODO Add support for structlog logger.bind [how does it even work?]
        raise ValueError("TODO Add support for structlog logger.bind [how does it even work?]")
    cloudpickle.CloudPickler.dispatch[structlog._config.BoundLoggerLazyProxy] = save_structlog_BoundLoggerLazyProxy
    cloudpickle.CloudPickler.dispatch[structlog.BoundLoggerBase] = save_structlog_BoundLoggerBase


def init_potoo():
    """cf. notebooks/__init__.py"""

    from potoo.python import ensure_python_bin_dir_in_path, install_sigusr_hooks
    # ensure_python_bin_dir_in_path()
    # install_sigusr_hooks()

    from potoo.pandas import set_display_on_sigwinch, set_display
    # set_display_on_sigwinch()
    set_display()

    from potoo.ipython import disable_special_control_backslash_handler, set_display_on_ipython_prompt, ipy_formats
    # disable_special_control_backslash_handler()
    # set_display_on_ipython_prompt()
    ipy_formats.set()

    from potoo.plot import plot_set_defaults
    plot_set_defaults()
