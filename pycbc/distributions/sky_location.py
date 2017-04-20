# Copyright (C) 2016  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""This modules provides classes for evaluating sky distributions in
right acension and declination.
"""


import pycbc.distributions
from pycbc.distributions import angular
from pycbc import detector
from pycbc.distributions.bounded import VARARGS_DELIM

class UniformSky(angular.UniformSolidAngle):
    """A distribution that is uniform on the sky. This is the same as
    UniformSolidAngle, except that the polar angle varies from pi/2 (the north
    pole) to -pi/2 (the south pole) instead of 0 to pi. Also, the default
    names are "dec" (declination) for the polar angle and "ra" (right
    ascension) for the azimuthal angle, instead of "theta" and "phi".
    """
    name = 'uniform_sky'
    _polardistcls = angular.CosAngle
    _default_polar_angle = 'dec'
    _default_azimuthal_angle = 'ra'


class SkyFromArrivalTimes(object):
    r"""A distribution in ra, dec and tc based on distributions on the arrival
    times in each detector.

    Parameters
    ----------
    tc_distributions : dict
        Dictionary of `detector name -> distribution` giving the distribution
        to use for each detector's arrival time.
    tc_ref_frame : str
        The detector to use for the coalescence time. Must be one of the
        detectors in the `tc_distributions` dict.
    """
    name = 'sky_from_ts'
    _params = ['ra', 'dec', 'tc'] 

    def __init__(self, tc_distributions, tc_ref_frame):
        super(SkyFromArrivalTimes, self).__init__()
        self.sky_distribution = UniformSky()
        self.t_distributions = tc_distributions
        # check that each distribution only has one parameter, and store
        self.t_params = {}
        for detname, distr in tc_distributions.items():
            if len(distr.params) != 1:
                raise ValueError("all tc distributions must only have one "
                                 "parameter")
            self.t_params[detname] = distr.params[0]
        # initialize all needed detectors
        self.detectors = {d: detector.Detector(d)
                          for d in tc_distributions.keys()}
        self.tc_ref_frame = tc_ref_frame
        try:
            self.tc_distribution = self.t_distributions.pop(tc_ref_frame)
        except KeyError:
            raise ValueError("must provide a tc distribution for {}".format(
                             tc_ref_frame))
        self.ref_detector = self.detectors.pop(tc_ref_frame)

    @property
    def params(self):
        return self._params

    def apply_boundary_conditions(self, **kwargs):
        """Applies the boundary conditions from all of the distributions in
        self.
        """
        # sky location
        kwargs.update(self.sky_distribution.apply_boundary_conditions(
            **kwargs))
        # tc
        kwargs.update(self.tc_distribution.apply_boundary_conditions(**kwargs))
        # the arrival times
        for distr in self.t_distributions.values():
            kwargs.update(distr.apply_boundary_conditions(**kwargs))
        return kwargs

    def __contains__(self, params):
        return all([params in dist for dist in self.t_distributions.values() +
                        [self.tc_distribution, self.sky_distribution]])

    def _logpdf(self, **kwargs):
        try:
            ra = kwargs['ra']
            dec = kwargs['dec']
            tc = kwargs['tc']
        except KeyError:
            raise ValueError("must provide an ra, dec, and tc")
        # compute the arrival time in each detector for this sky location
        for detname, det in self.detectors.items():
            det_tc =  tc - self.ref_detector.time_delay_from_detector(det,
                ra, dec, tc)
            kwargs.update({self.t_params[detname]: det_tc})
        if kwargs not in self:
            return -numpy.inf
        lp = self.tc_distribution.logpdf(**kwargs)
        # add the pdfs from all of the other times
        for distr in self.t_distributions.values():
            lp += distr.logpdf(**kwargs)
        return lp

    def _pdf(self, **kwargs):
        return numpy.exp(self.logpdf(**kwargs))
            
    def __call__(self, **kwargs):
        return self.logpdf(**kwargs)

    def rvs(self, size=1, param=None):
        """Random variates are not implemented for this distribution."""
        raise NotImplementedError("Random variates are not implemented for "
                                  "this distribution.")


    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file.
        
        The variable args must contain 'ra+dec+tc'. The section must specify
        a reference detector, and names of sections from which to retrieve
        distributions for each arrival time. Example:

        .. code::
            [{section}-ra+dec+tc]
            name = sky_from_ts
            H1 = tc_prior-tc
            L1 = tl_prior-tL
            tc-ref-frame = H1

            [tc_prior-tc]
            name = gaussian
            tc_mean = 11
            tc_var = 1e-5

            [tl_prior-tl]
            name = gaussian
            tl_mean = 12
            tl_var = 1e-5

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            `prior.VARARGS_DELIM`. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        SkyFromArrivalTimes
            A distribution instance from the pycbc.inference module.
        """
        tag = variable_args
        variable_args = variable_args.split(VARARGS_DELIM)
        if not set(variable_args) == set(cls._params):
            raise ValueError("Not all parameters used by this distribution "
                             "included in tag portion of section name")
        special_args = ["name", "tc-ref-frame"]
        tc_ref_frame = cp.get_opt_tag(section, 'tc-ref-frame', tag)
        # get the distribution for each arrival time
        tc_distributions = {}
        for detname in cp.options( "-".join([section,tag])):
            if detname in special_args:
                continue
            secname = cp.get_opt_tag(section, detname, tag)
            # get the name of the distribution and the variable name from the
            # section name that is pointed to
            secname = secname.split('-')
            varname = secname[-1]
            secname = '-'.join(secname[:-1])
            distname = cp.get_opt_tag(secname, 'name', varname)
            tc_distributions[detname] = \
                pycbc.distributions.distribs[distname].from_config(cp,
                secname, varname)
        return cls(tc_distributions, tc_ref_frame)

__all__ = ['UniformSky', 'SkyFromArrivalTimes']
