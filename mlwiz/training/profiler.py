import bisect
import datetime
import time

from mlwiz.training.event.handler import EventHandler


# Decorator class. Ideas taken from "Fluent Python" book
class Profiler:
    r"""
    A decorator class that is applied to a
    :class:`~mlwiz.training.event.handler.EventHandler` object implementing a
    set of callback functions. For each callback, the Profiler stores the
    average and total running time across epochs.
    When the experiment terminates (either correctly or abruptly) the
    Profiler can produce a report to be stored in the experiment's log file.

    The Profiler is used as a singleton, and it produces wrappers that update
    its own state.

    Args:
        threshold (float): used to filter out callback functions that consume
            a negligible amount of time from the report

    Usage:
        Istantiate a profiler, and then register an event_handler with the
        syntax profiler(event_handler), which returns
        another object implementing the
        :class:`~mlwiz.training.event.handler.EventHandler` interface
    """

    def __init__(self, threshold: float):
        # we filter out computation that takes a negligible amount of time from
        # the report (< threshold)
        self.threshold = threshold
        self.callback_elapsed = {}
        self.callback_calls = {}

    def __call__(self, event_handler: EventHandler) -> object:
        r"""
        Wraps a :class:`~mlwiz.training.event.handler.EventHandler` object,
        so that whenever one of its callbacks is triggered the Profiler
        can compute and store statistics about the time required to execute it.

        Args:
            event_handler
                (:class:`~mlwiz.training.event.handler.EventHandler`): the
                object implementing a subset of the callbacks defined in
                the :class:`~mlwiz.training.event.handler.EventHandler`
                interface

        Returns:
            an object that transparently updates the profiler's state
            whenever a callback is triggered
        """
        self.callback_elapsed[event_handler.__class__.__name__] = {}
        self.callback_calls[event_handler.__class__.__name__] = {}

        def clock(func):
            """
            Decorator that takes a function and "clocks" it, by taking
            the time needed to execute it and updating
            statistics of the :obj:`Profiler` instance.
            """

            def clocked(*args, **kwargs):
                """
                Monitors the time to execute a callback function of some
                'obj':`EventHandler`.
                """
                class_name = event_handler.__class__.__name__
                callback_name = func.__name__
                t0 = time.time()
                # print(f'Callback {cls.__class__.__name__}: calling
                # {func.__name__} with args {args} and kwargs {kwargs}')
                result = func(*args, **kwargs)
                elapsed = time.time() - t0

                if callback_name in self.callback_elapsed[class_name]:
                    self.callback_elapsed[class_name][callback_name] += elapsed
                    self.callback_calls[class_name][callback_name] += 1
                else:
                    self.callback_elapsed[class_name][callback_name] = elapsed
                    self.callback_calls[class_name][callback_name] = 1

                return result

            return clocked

        class getattribute(type):
            """
            Taken from: (stackoverflow.com/questions/3155436/
            getattr-for-static-class-variables-in-python)
            We need to do this to invoke __getattr__ on a class
            (stackoverflow.com/questions/100003/what-are-metaclasses-in-python)
            TLDR We want the class ClockedCallback (which is an object itself)
            to have a method __getattr__. if we specified __getattr__ INSIDE
            ClockedCallback, rather than in its metaclass,
            the class ClockedCallback would not use the overridden __getattr__
            """

            def __getattr__(self, name):
                """
                See also class description. The method clocks the callback
                function whenever a specific attribute has to be retrieved.
                """
                return clock(
                    getattr(event_handler, name)
                )  # needed to implement callback calls with generic names

        # Relies on closures
        class ClockedCallback(metaclass=getattribute):
            """
            Dummy class to override method `__getattr__`
            """

            pass

        return ClockedCallback

    def report(self) -> str:
        r"""
        Builds a report string containing the statistics of the
        experiment accumulated so far.

        Returns:
            a string containing the report
        """
        total_time_experiment = 0.0
        profile_str = f'{"*" * 25} Profiler {"*" * 25} \n \n'
        profile_str += f"Threshold: {self.threshold} \n \n"

        for class_name, v in self.callback_elapsed.items():
            sorted_avg_elapsed = []

            for callback_name, v1 in v.items():
                n = self.callback_calls[class_name][callback_name]

                # Depending on when a KeyboardInterrupt is triggered
                if n == 0:
                    continue

                avg_elapsed = v1 / n

                if avg_elapsed > self.threshold:
                    bisect.insort(
                        sorted_avg_elapsed, (avg_elapsed, v1, callback_name)
                    )

            # Release resources
            v.clear()

            profile_str += f"{class_name} \n \n"

            for avg_elapsed, total_elapsed, callback_name in reversed(
                sorted_avg_elapsed
            ):
                total_time_experiment += total_elapsed
                seconds_elapsed = datetime.timedelta(seconds=total_elapsed)
                profile_str += (
                    f"\t {callback_name} --> "
                    f"Avg: {avg_elapsed} s, "
                    f"Total: {str(seconds_elapsed)} \n"
                )
            profile_str += "\n"

        total_time = datetime.timedelta(seconds=total_time_experiment)
        profile_str += f"Total time of the experiment: {str(total_time)} \n \n"
        profile_str += f'{"*" * 60}'
        return profile_str
