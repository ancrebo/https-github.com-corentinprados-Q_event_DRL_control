#!/bin/env python
#
# SATELLITE TOOLS
#
# Chrono module for performance profiling.
#
# Arnau Miro, Elena Terzic
from __future__ import print_function, division

import numpy as np, time as time_module

from logging_config import configure_logger, DEFAULT_LOGGING_LEVEL

# Set up logger
logger = configure_logger(__name__, default_level=DEFAULT_LOGGING_LEVEL)

logger.info("%s.py: Logging level set to %s", __name__, logger.level)

CHANNEL_DICT = {}


class channel(object):
    """
    This is a channel for the cr counter
    """

    def __init__(self, name, tmax, tmin, tsum, nop, tini):
        self._name = name  # Name of the channel
        self._tmax = tmax  # Maximum time of the channel
        self._tmin = tmin  # Minimum time of the channel
        self._tsum = tsum  # Total time of the channel
        self._nop = nop  # Number of operations
        self._tini = (
            tini  # Initial instant (if == 0 channel is not being take into account)
        )
        logger.debug("cr.channel.init: channel created with name %s", name)

    def __str__(self):
        return "name %-25s n %4d tmin %e tmax %e tavg %e tsum %e" % (
            self.name,
            self.nop,
            self.tmin,
            self.tmax,
            self.tavg,
            self.tsum,
        )

    def __add__(self, other):
        new = copy.deepcopy(self)
        new._tmax = max(new._tmax, other._tmax)
        new._tmin = min(new._tmin, other._tmin)
        new._tsum += other._tsum
        new._nop += other._nop
        return new

    def __iadd__(self, other):
        self._tmax = max(self._tmax, other._tmax)
        self._tmin = min(self._tmin, other._tmin)
        self._tsum += other._tsum
        self._nop += other._nop
        return self

    def reset(self):
        """
        Reset the channel
        """
        logger.debug(
            "cr.channel.reset: starting `reset` method for channel %s", self.name
        )
        self._tmax = 0.0
        self._tmin = 0.0
        self._tsum = 0.0
        self._nop = 0.0
        self._tini = 0.0
        logger.debug(
            "cr.channel.reset: finished `reset` method for channel %s", self.name
        )

    def restart(self):
        logger.debug(
            "cr.channel.restart: starting `restart` method for channel %s", self.name
        )
        self._tini = 0.0
        logger.debug(
            "cr.channel.restart: finished `restart` method for channel %s", self.name
        )

    def start(self, tini):
        logger.debug(
            "cr.channel.start: starting `start` method for channel %s", self.name
        )
        self._tini = tini
        logger.debug(
            "cr.channel.start: finished `start` method for channel %s", self.name
        )

    def increase_nop(self):
        logger.debug(
            "cr.channel.increase_nop: starting `increase_nop` method for channel %s",
            self.name,
        )
        self._nop += 1
        logger.debug(
            "cr.channel.increase_nop: finished `increase_nop` method for channel %s",
            self.name,
        )

    def increase_time(self, time):
        logger.debug("cr.channel.increase_time: starting `increase_time` method")
        self._tsum += time
        logger.debug("cr.channel.increase_time: finished `increase_time` method")

    def set_max(self, time):
        logger.debug("cr.channel.set_max: starting `set_max` method")
        if time > self._tmax or self._nop == 1:
            self._tmax = time
        logger.debug("cr.channel.set_max: finished `set_max` method")

    def set_min(self, time):
        logger.debug("cr.channel.set_min: starting `set_min` method")
        if time < self._tmin or self._nop == 1:
            self._tmin = time
        logger.debug("cr.channel.set_min: finished `set_min` method")

    def elapsed(self, time):
        logger.debug("cr.channel.elapsed: starting `elapsed` method")
        logger.debug("cr.channel.elapsed: finished `elapsed` method")
        return time - self._tini

    def is_running(self):
        logger.debug("cr.channel.is_running: starting `is_running` method")
        logger.debug("cr.channel.is_running: finished `is_running` method")
        return not self._tini == 0

    @classmethod
    def new(cls, name):
        """
        Create a new channel
        """
        logger.debug("cr.channel.new: starting `new` method")
        logger.debug("cr.channel.new: finished `new` method")
        return cls(name, 0, 0, 0, 0, 0)

    @property
    def name(self):
        logger.debug("cr.channel.name: starting `name` method")
        logger.debug("cr.channel.name: finished `name` method")
        return self._name

    @property
    def nop(self):
        logger.debug("cr.channel.nop: starting `nop` method")
        logger.debug("cr.channel.nop: finished `nop` method")
        return self._nop

    @property
    def tmin(self):
        logger.debug("cr.channel.tmin: starting `tmin` method")
        logger.debug("cr.channel.tmin: finished `tmin` method")
        return self._tmin

    @property
    def tmax(self):
        logger.debug("cr.channel.tmax: starting `tmax` method")
        logger.debug("cr.channel.tmax: finished `tmax` method")
        return self._tmax

    @property
    def tavg(self):
        logger.debug("cr.channel.tavg: starting `tavg` method")
        logger.debug("cr.channel.tavg: finished `tavg` method")
        return self._tsum / (1.0 * self._nop) if self._nop > 0 else 0.0

    @property
    def tsum(self):
        logger.debug("cr.channel.tsum: starting `tsum` method")
        logger.debug("cr.channel.tsum: finished `tsum` method")
        return self._tsum

    @property
    def report(self):
        logger.debug("cr.channel.report: starting `report` method")
        logger.debug("cr.channel.report: finished `report` method")
        return np.array([self.nop, self.tmin, self.tmax, self.tavg, self.tsum])


def _newch(ch_name):
    """
    Add a new channel to the list
    """
    logger.debug("cr._newch: starting `_newch` method")
    CHANNEL_DICT[ch_name] = channel.new(ch_name)
    logger.debug("cr._newch: finished `_newch` method")
    return CHANNEL_DICT[ch_name]


def _findch(ch_name):
    """
    Look for the channel
    """
    logger.debug("cr._findch: starting `_findch` method")
    logger.debug("cr._findch: finished `_findch` method")
    return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else None


def _addsuff(ch_name, suff=-1):
    logger.debug("cr._addsuff: starting `_addsuff` method")
    logger.debug("cr._addsuff: finished `_addsuff` method")
    return ch_name if suff <= 0 else "%s%02d" % (ch_name, suff)


def _findch_crash(ch_name):
    """
    Look for the channel and crash if it does not exist
    """
    logger.debug("cr._findch_crash: starting `_findch_crash` method")
    if not ch_name in CHANNEL_DICT.keys():
        raise ValueError("Channel %s does not exist!" % ch_name)
    logger.debug("cr._findch_crash: finished `_findch_crash` method")
    return CHANNEL_DICT[ch_name]


def _findch_create(ch_name):
    """
    Find the channel and if not found create it
    """
    logger.debug("cr._findch_create: starting `_findch_create` method")
    logger.debug("cr._findch_create: finished `_findch_create` method")
    return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else _newch(ch_name)


def _gettime():
    """
    Returns the number of second since an arbitrary instant but fixed.
    Returned value will always be > 0.
    """
    logger.debug("cr._gettime: starting `_gettime` method")
    logger.debug("cr._gettime: finished `_gettime` method")
    return time_module.time()


def _info_serial():
    logger.debug("cr._info_serial: starting `_info_serial` method")
    tsum_array = np.array([CHANNEL_DICT[key].tsum for key in CHANNEL_DICT.keys()])
    name_array = np.array([CHANNEL_DICT[key].name for key in CHANNEL_DICT.keys()])

    ind = np.argsort(tsum_array)  # sorted indices

    print("\ncr_info:")
    for ii in ind[::-1]:
        print(CHANNEL_DICT[name_array[ii]])
    print("")
    logger.debug("cr._info_serial: finished `_info_serial` method")


def _report_serial(fname):
    logger.debug("cr._report_serial: starting `_report_serial` method")
    tsum_array = np.array([CHANNEL_DICT[key].tsum for key in CHANNEL_DICT.keys()])
    name_array = np.array([CHANNEL_DICT[key].name for key in CHANNEL_DICT.keys()])

    ind = np.argsort(tsum_array)  # sorted indices

    file = open(fname, "w")
    # Header
    file.write("# name, n, tmin, tmax, tavg, tsum\n")

    for ii in ind[::-1]:
        r = CHANNEL_DICT[name_array[ii]].report
        file.write(
            "%-25s, %4d, %e, %e, %e, %e\n"
            % (name_array[ii], r[0], r[1], r[2], r[3], r[4])
        )
    logger.debug("cr._report_serial: finished `_report_serial` method")


def cr_reset():
    """
    Delete all channels and start again
    """
    logger.debug("cr.cr_reset: starting `cr_reset` method")

    CHANNEL_DICT = {}

    logger.debug("cr.cr_reset: finished `cr_reset` method")


def cr_info(rank=-1):
    """
    Print information - order by major sum
    """
    logger.debug("cr.cr_info: starting `cr_info` method")

    _info_serial()

    logger.debug("cr.cr_info: finished `cr_info` method")


def cr_report(filename):
    """
    Print a report of the execution times in a file
    """
    logger.debug("cr.cr_report: starting `cr_report` method")

    _report_serial(filename)

    logger.debug("cr.cr_report: finished `cr_report` method")


def cr_start(ch_name, suff):
    """
    Start the chrono of a channel
    """
    logger.debug("cr.cr_start: starting `cr_start` method")
    name_tmp = _addsuff(ch_name, suff)
    channel = _findch_create(name_tmp)
    if channel.is_running():
        raise ValueError("Channel %s was already set!" % channel.name)
    channel.start(_gettime())
    logger.debug("cr.cr_start: finished `cr_start` method")


def cr_stop(ch_name, suff):
    """
    Stop the chrono of a channel
    """
    logger.debug("cr.cr_stop: starting `cr_stop` method")
    end = _gettime()
    name_tmp = _addsuff(ch_name, suff)
    channel = _findch_crash(name_tmp)
    time = channel.elapsed(end)

    channel.increase_nop()
    channel.set_max(time)
    channel.set_min(time)
    channel.increase_time(time)

    channel.restart()
    logger.debug("cr.cr_stop: finished `cr_stop` method")


def cr_time(ch_name, suff):
    """
    Get the time of a channel that is running; channel keeps running
    """
    logger.debug("cr.cr_time: starting `cr_time` method")
    end = _gettime()
    name_tmp = _addsuff(ch_name, suff)
    channel = _findch_crash(name_tmp)
    logger.debug("cr.cr_time: finished `cr_time` method")
    return channel.elapsed(end)
