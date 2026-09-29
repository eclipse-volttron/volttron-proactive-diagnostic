# -*- coding: utf-8 -*- {{{
# ===----------------------------------------------------------------------===
#
#                 Installable Component of Eclipse VOLTTRON
#
# ===----------------------------------------------------------------------===
#
# Copyright 2022 Battelle Memorial Institute
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# ===----------------------------------------------------------------------===
# }}}

import gevent
import importlib.metadata
import logging
import sys

from collections import defaultdict
from sympy import symbols
from sympy.logic.boolalg import BooleanFalse, BooleanTrue
from sympy.parsing.sympy_parser import parse_expr


from importlib.metadata import distribution, PackageNotFoundError
try:
    distribution('volttron-core')
    from volttron.client.messaging import topics
    from volttron.client.vip.agent import Agent
    from volttron.utils import format_timestamp, get_aware_utc_now, load_config, vip_main
    from volttron.utils.jsonrpc import RemoteError
    from volttron.utils.math_utils import mean
    from volttron.utils.scheduling import cron
except PackageNotFoundError:
    from volttron.platform.agent.utils import load_config, setup_logging, vip_main
    from volttron.platform.agent.utils import format_timestamp, get_aware_utc_now
    from volttron.platform.scheduling import cron
    from volttron.platform.messaging import topics
    from volttron.platform.agent.math_utils import mean
    from volttron.platform.vip.agent import Agent
    from volttron.platform.jsonrpc import RemoteError

__version__ = importlib.metadata.version("mypackage")

setup_logging()
_log = logging.getLogger(__name__)


class Diagnostic:
    """The ProactiveDiagnostics class can be configured to instantiate
    multiple Diagnostics. Each diagnostic potentially has multiple control
    steps each with fault detection rule(s) to evaluate.
    """
    def __init__(self, config, parent):
        """Each diagnostic is instantiated by the ProactiveDiagnostic.
        config dictionary contains proactive control information and fault
        detection rules to evaluate.

        :param parent:  ProactiveDiagnostic object
        :param config: configuration dictionary for diagnostic
        :return: None
        """
        self.parent = parent
        self.vip = parent.vip
        self.name = config.get("name")
        # This facilitates performing diagnostic on a remote platform.
        # If connecting to a remote platform this value should be
        # set to the string name of the remote as configured in
        # the external_platform_discovery.json file
        # in VOLTTRON_HOME (~/.volttron by defualt).
        self.remote = parent.remote_platform
        _log.debug("Configure: %s", self.name)
        self.control_parameters = config.get("control")
        _log.debug("Configure control: %s", self.control_parameters)
        # The fault_code can be a string or number
        # associated with a fault for the diagnostic.
        self.fault_code = config.get("fault_code")
        # The non_fault_code can be a string or number
        # associated with no fault detected for the diagnostic.
        self.non_fault_code = config.get("non_fault_code")
        # The fault_condition variable can be set to "all", meaning
        # that all conditions must evaluate to True for a positive fault
        # detection or "any" meaning that any condition that evaluates to
        # can evaluate to True for a positive fault detection.
        self.fault_condition = config.get("fault_condition", "all")
        self.evaluations = []
        self.analysis_topic = ["/".join(["record", device, self.name])
                               for device in self.parent.base_rpc_path]
        self.analysis_message = {
            "result": None
        }
        self.headers = {}

    def run(self):
        """Main run method for each diagnostic in the ProactiveDiagnostics
        diagnostic list.  This method is triggered based on the cron schedule
        set by the ProactiveDiagnostics run_schedule parameter.
        :return: None
        """
        _log.debug("Run diagnostic: %s", self.name)
        # Each diagnostic may have multiple control steps
        # in the proactive diagnostic process.
        for diagnostic in self.control_parameters:
            revert_action = {}
            # Point name to set based on parents
            # base_rpc_path (campus/building/device)
            control_points = diagnostic.get("points")
            # Time to wait after control action prior
            # to performing data analysis.
            steady_state_interval = diagnostic.get("steady_state_interval")
            for device in self.parent.base_rpc_path:
                for point, value in control_points.items():
                    point_to_set = device(point=point)
                    # revert action restore will cause diagnostic to store
                    # value of point prior to control.  This value will be
                    # used to restore the device to normal operations.
                    # If revert_action != restore then it is assumed the the
                    # device can be released using None (BACnet device).
                    if self.parent.revert_action == "restore":
                        _log.debug("Using get point to obtain restore value!")
                        value = self.vip.rpc.call(
                            self.parent.actuator,
                            "get_point",
                            "proactive",
                            point_to_set).get(timeout=5)
                        revert_action[point_to_set] = value
                    else:
                        _log.debug("Using release to write None!")
                        revert_action[point_to_set] = None
                    try:
                        _log.debug("Control: %s - "
                                  "value: %s", point_to_set, value)
                        result = self.vip.rpc.call(
                            self.parent.actuator,
                            "set_point",
                            "proactive",
                            point_to_set,
                            value,
                            priority=8).get(timeout=5)
                        _log.debug("Actuator %s "
                                  "result %s", point_to_set, result)
                    except RemoteError as ex:
                        _log.warning("Failed to set point %s"
                                    "(RemoteError): %s", point_to_set, str(ex))
                        self.restore(revert_action)
                        return
            _log.debug("Control steady state: %s", steady_state_interval)
            # Sleep and allow steady state conditions to be achieved.
            gevent.sleep(steady_state_interval)
            # 10 data points will be queried for analysis
            # over the data_collection_interval set in the
            # diagnostics configuration.
            data_query_interval = \
                int(diagnostic.get("data_collection_interval")/10)
            try:
                self.analysis(diagnostic["analysis"], data_query_interval)
            except KeyError as ex:
                _log.warning("Diagnostic name: %s -- analysis dictionary "
                            "is missing for diagnostic - %s", self.name, ex)
        self.report()
        self.restore(revert_action)

    def analysis(self, analysis_parameters, data_collection_interval):
        """  Evaluate fault detection rules based on data queried from
        the actuator agent's get_point method.

        :param analysis_parameters: dictionary with data points
        and rules to evaluate for fault detection

        :param data_collection_interval: time to sleep between
        collection of data for analysis

        :return: None
        """
        data = defaultdict(list)
        # list of point name to get based on parents
        # base_rpc_path (campus/building/device)
        operation_args = analysis_parameters.get("points")
        symbols(operation_args)
        # list of rules to evaluate for fault detection.
        rules = analysis_parameters.get("rule_list")
        inconclusive = analysis_parameters.get("inconclusive_conditions_list")
        if rules is None or not rules:
            _log.warning("Diagnostic name: %s is missing rule to evaluate "
                        "fault condition check configuration file!", self.name)
            return
        if not all(isinstance(rule, str) for rule in rules):
            _log.warning("Rule for diagnostic name %s must be a string, '"
                        "fix configuration!", self.name)
            return
        rule_list = [parse_expr(op) for op in rules]
        if inconclusive is not None and inconclusive:
            inconclusive_list = [parse_expr(op) for op in inconclusive]
        else:
            inconclusive_list = []
        for _ in range(10):
            for operation_arg in operation_args:
                for device in self.parent.base_rpc_path:
                    point_to_get = device(point=operation_arg)
                    value = self.vip.rpc.call(
                        self.parent.actuator,
                        "get_point",
                        point_to_get).get(timeout=5)
                    data[operation_arg].append(value)
            gevent.sleep(data_collection_interval)
        rule_data = []
        for key, value in data.items():
            rule_data.append((key, mean(value)))
        _log.debug("Diagnostic data : %s", rule_data)
        # Support for multi-condition fault detection
        if self.inconclusive_diagnostic_check(inconclusive_list,
                                              rule_data):
            self.evaluations.append(-1)
            return

        results = [rule.subs(rule_data) for rule in rule_list]
        _log.debug("Results type : %s", type(results[0]))
        # Verify that all items in results evaluate to True or False.
        # Incorrectly named points or improper sympy syntax could
        # result in this occurring
        # https://docs.sympy.org/latest/modules/parsing.html
        if not all(isinstance(evaluation, (BooleanFalse, BooleanTrue))
                   for evaluation in results):
            _log.warning("Evaluation did not produce True or False "
                        "required for indicating fault/no-fault")
            _log.warning("Check sympy syntax in the analysis rule_list "
                        "and verify that data is available in the VOLTTRON "
                        "driver for that device/point")
            result = False
        else:
            result = False not in results
        self.evaluations.append(result)
        _log.debug("Analysis result : %s", result)

    @staticmethod
    def inconclusive_diagnostic_check(inconclusive_list, data):
        """Verify individual diagnostic prerequisites are met.

        :param inconclusive_list: list of sympy expressions
        :param data: list of key value pairs for data to evaluate
        sympy expressions

        :return: returns False if
        """
        # If this list is empty then there are no
        # conditions that could lead to an inconclusive diagnostic
        if not inconclusive_list:
            return False
        _log.debug("Diagnostic prerequisites : %s", inconclusive_list)
        # Evaluate each condition with the device data
        inconclusive_results = [con.subs(data) for con in inconclusive_list]
        _log.debug("Evaluation of prerequisites : %s", inconclusive_results)
        # Verify that all the conditions evaluated to booleans.
        # A non-boolean value means there was a problem evaluating the
        # sympy expression.
        if not all(isinstance(evaluation, (BooleanFalse, BooleanTrue))
                   for evaluation in inconclusive_results):
            _log.warning("Inconclusive checks did not produce "
                        "True or False data type: %s",
                        type(inconclusive_list[0]))
            _log.warning("Check sympy syntax in the analysis dict for "
                        "inconclusive_conditions_list.  Verify "
                        "that data is available in the VOLTTRON "
                        "driver for that device/point")
            return True
        return False in inconclusive_results

    def report(self):
        """ Report result of diagnostic analysis and publish
        to the VOLTTRON message bus.

        :return: None
        """
        # Multiple control steps and analysis can occur for each diagnostic
        # if self.fault_condition == all then all steps must have a fault
        # condition for a fault to be reported.
        self.headers = {
            "Date": format_timestamp(get_aware_utc_now()),
            "Timestamp": format_timestamp(get_aware_utc_now())
        }
        analysis = {}
        if -1 in self.evaluations:
            _log.debug("Diagnostic %s resulted in inconclusive result",
                      self.name)
            analysis = {"result": -1}
            for publish_topic in self.analysis_topic:
                self.vip.pubsub.publish("pubsub",
                                        self.analysis_topic,
                                        headers=self.headers,
                                        message=analysis)
            self.evaluations = []
            return

        if self.fault_condition == "any":
            if False in self.evaluations:
                _log.debug("%s - no fault detected", self.name)
                analysis = {"result": self.non_fault_code}
            else:
                _log.debug("%s - fault detected", self.name)
                analysis = {"result": self.fault_code}
        # Multiple control steps and analysis can occur for each diagnostic
        # if self.fault_condition == "any"" then any step where a
        # fault condition is detected will lead to reporting a fault.
        else:
            if True in self.evaluations:
                _log.debug("%s - fault detected", self.name)
                analysis = {"result": self.fault_code}
            else:
                _log.debug("%s - no fault detected", self.name)
                analysis = {"result": self.non_fault_code}

        # Reinitialize evaluations list for use in next diagnostic run.
        for publish_topic in self.analysis_topic:
            self.vip.pubsub.publish("pubsub",
                                    publish_topic,
                                    headers=self.headers,
                                    message=analysis)
        self.evaluations = []

    def restore(self, revert_action):
        """ Restore device to normal operations.

        :param revert_action: dictionary of point topics
        and values to call actuator agent's set_point method.

        :return: None
        """
        for point_to_set, value in revert_action.items():
            try:
                _log.debug("Revert control for "
                          "%s with value %s", point_to_set, value)
                result = self.vip.rpc.call(
                    self.parent.actuator,
                    "set_point",
                    "proactive",
                    point_to_set,
                    value, priority=8).get(timeout=5)
                _log.debug("Actuator %s "
                          "result %s", point_to_set, result)
            except RemoteError as ex:
                _log.warning("Failed to revert point "
                            "%s (RemoteError): %s", point_to_set, str(ex))
                continue


class ProactiveDiagnostics(Agent):
    """This application allows for highly customizable
    proactive fault detection of building systems.

    """
    def __init__(self, config_path, **kwargs):
        """Setup default_config either from config file or generic defaults
        if file is not provided.  Setup VOLTTRON config store callback for
        using and updating configuration from store.

        :param kwargs: empty
        :return: None
        """
        super(ProactiveDiagnostics, self).__init__(**kwargs)
        # If a configuration file is used the agent will utilize
        # this for the default configuration.  If a config is in
        # config store this will override the config file.
        file_config = load_config(config_path)
        default_config = {
            "campus": "campus",
            "actuator": "platform.actuator",
            "building": "building",
            "device": ["device"],
            "prerequisites": {},
            "run_schedule": "* * * * *",
            "diagnostics": [],
        }
        if file_config:
            self.default_config = file_config
        else:
            self.default_config = default_config
        self.run_schedule = None
        self.revert_action = "release"
        self.base_rpc_path = []
        self.device_topics_list = []
        self.diagnostics = None
        self.diagnostics_container = []
        self.prerequisites_expr_list = []
        self.prerequisites_data_required = {}
        self.prerequisites_variables = None
        self.remote_platform = None
        # Add default config to store.

        self.vip.config.set_default("config", self.default_config)
        # Add callback for "New" or "UPDATE" to config in config store
        self.vip.config.subscribe(self.configure_main,
                                  actions=["NEW", "UPDATE"],
                                  pattern="config")

    def configure_main(self, config_name, action, contents):
        """This triggers configuration of the ProactiveDiagnostic via
        the VOLTTRON configuration store.

        :param config_name: canonical name is config

        :param action: on instantiation this is "NEW" or
        "UPDATE" if user uploads update config to store

        :param contents: configuration contents

        :return: None
        """
        _log.debug("Update %s for %s", config_name, self.core.identity)
        config = self.default_config.copy()
        config.update(contents)
        if action == "NEW" or "UPDATE":
            # The run schedule should be a cron string
            # https://volttron.readthedocs.io/en/develop/devguides/agent_development/Agent-Development-Cheatsheet.html
            # https://crontab.guru/
            self.run_schedule = config.get("run_schedule")
            # The campus, building, device parameters are used to build the
            # (devices/campus/building/device/all) subscription for device data
            # coming from master driver and the rpc to do actuation
            # (campus/building/device/point)
            campus = config.get("campus", "")
            building = config.get("building", "")
            device_list = config.get("device", [])
            self.revert_action = config.get("revert_action", "release")
            # Configure global diagnostic prerequisites.
            # Data mechanism is through subscription.
            # Evaluation is only done prior to running diagnostic.
            prerequisites = config.get("prerequisites", {})
            self.actuator = config.get("actuator_vip", "platform.actuator")
            self.remote_platform = config.get("remote_platform")

            self.base_rpc_path = []
            self.device_topics_list = []
            if not device_list:
                _log.warning("Configuration ERROR: no device_list "
                            "configured for diagnostic!")
                _log.warning("Check configuration and update "
                            "device_list!")

            for device in device_list:
                self.base_rpc_path.append(
                    topics.RPC_DEVICE_PATH(campus=campus,
                                           building=building,
                                           unit=device,
                                           path="",
                                           point=None))
                self.device_topics_list.append(topics.DEVICES_VALUE(
                    campus=campus, building=building,
                    unit=device, path="", point="all"))

            diagnostics = config.get("diagnostics", [])
            if not diagnostics:
                _log.warning("Configuration ERROR diagnostics"
                            "information is not configured!")
                _log.warning("Diagnostic cannot be performed, "
                            "Update configuration!")

            self.diagnostics = diagnostics
            self.diagnostics_container = []
            self.prerequisites_expr_list = []
            self.prerequisites_data_required = {}
            self.prerequisites_variables = None
            if prerequisites:
                self.initialize_prerequisites(prerequisites)
            else:
                _log.debug("No diagnostic prerequisites configured!")
            self.starting_base()

    def starting_base(self, **kwargs):
        """Instantiate each diagnostic in the diagnostic list for the
        ProactiveDiagnostic.  Setup device data subscriptions.

        :param: kwargs: empty
        :return: None
        """
        # For each diagnostic instantiate a Diagnostic
        # and pass it a configuration (diagnostic) and a reference to the
        # ProactiveDiagnostic.
        for diagnostic in self.diagnostics:
            _log.debug("Configure %s", diagnostic.get("name"))
            self.diagnostics_container.append(Diagnostic(diagnostic, self))
        for device in self.device_topics_list:
            _log.debug("Subscribing to %s", device)
            self.vip.pubsub.subscribe(peer="pubsub",
                                      prefix=device,
                                      callback=self.new_data,
                                      all_platforms=True)
        # Using cron string in configuration schedule the diagnostics to run.
        self.core.schedule(cron(self.run_schedule), self.run_process)

    def run_process(self):
        """Main run process for ProactiveDiagnostics.  This is function
        is triggered  based on the cron schedule in run_schedule.
        Loops though all instantiated diagnostics and run them.

        :return: None
        """
        if not self.check_prerequisites():
            _log.debug("Prerequisites not met!")
        else:
            _log.debug("Prerequisites met!")
            # Call each Diagnostic instance run method.
            for diagnostic in self.diagnostics_container:
                diagnostic.run()

    def initialize_prerequisites(self, prerequisites):
        """Initialize and store information associated with evaluation
        of the diagnostic prerequisites.

        :param prerequisites: dictionary with information associated
        with diagnostic prerequisites.

        :return: None
        """
        # list of point name associated with diagnostic prerequisites
        # data is recieved in new_data method and subscriptions to device
        # data are made in starting_base
        prerequisites_args = prerequisites.get("condition_args")
        # List of rules to evaluate to determine if conditions permit
        # running the proactive diagnostics.
        prerequisites_list = prerequisites.get("conditions")
        for point in prerequisites_args:
            self.prerequisites_data_required[point] = []
        self.prerequisites_variables = symbols(prerequisites_args)
        for prerequisite in prerequisites_list:
            self.prerequisites_expr_list.append(parse_expr(prerequisite))

    def check_prerequisites(self):
        """Evaluate realtime data to determine if the diagnostic
        prerequisites are met.  If this function returns False the
        diagnostics in the diagnostic list will not run.

        :return: bool
        """
        if self.prerequisites_data_required:
            avg_data = []
            prerequisite_eval_list = []
            for key, value in self.prerequisites_data_required.items():
                try:
                    avg_data.append((key, mean(value)))
                except ValueError as ex:
                    _log.warning("Exception prerequisites %s", ex)
                    _log.warning("Not enough data to verify "
                                "diagnostic prerequisites: %s", key)
                    return False
            for condition in self.prerequisites_expr_list:
                evaluation = condition.subs(avg_data)
                prerequisite_eval_list.append(evaluation)
                _log.debug("Prerequisite %s evaluation: %s",
                          condition, evaluation)
            if prerequisite_eval_list:
                if False in prerequisite_eval_list:
                    return False
                return True
        return True

    def new_data(self, peer, sender, bus, topic, headers, message):
        """Call back for new data to track the diagnostic prerequisites.
        In the future this function could be employed to do passive
        diagnostics as well.

        :param peer:
        :param sender:
        :param bus:
        :param topic: string - devices/campus/building/device/all
        :param headers:
        :param message: array of key value pairs for data from
        VOLTTRON master driver and metadata

        :return:
        """
        # topic of form:  devices/campus/building/device
        _log.info("Data Received for %s", topic)

        data = message[0]
        for point in self.prerequisites_data_required:
            if point in data:
                _log.debug("Point %s - for device - %s added "
                          "to prerequisites array.", point, topic)
                self.prerequisites_data_required[point].append(data[point])
                # keep track of last 5 measurements for evaluation of
                # diagnostic global prerequisites
                self.prerequisites_data_required[point] = \
                    self.prerequisites_data_required[point][-5:]
                _log.debug("Prerequisite data %s.",
                          self.prerequisites_data_required)
            else:
                _log.warning("Possible Prerequisite data configuration error!")
                _log.warning("Point - %s for device - %s not available. "
                            "Check configuration", point, topic)


def main():
    """Main method called by the aip."""
    try:
        vip_main(ProactiveDiagnostics, version=__version__)
    except Exception as exception:
        _log.exception("unhandled exception")
        _log.error(repr(exception))


if __name__ == "__main__":
    # Entry point for script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
