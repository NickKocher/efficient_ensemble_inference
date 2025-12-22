from time import time
import logging

class TimerDisplay:
    """
    A simple utility class to display elapsed time for a given operation.
    """
    
    def __init__(self, rounds_total, n_digits_step=30, tabular = True, heading=None):
        self.rounds_total = rounds_total
        self.n_digits = len(str(rounds_total))
        self.n_digits_step = n_digits_step
        self.start_time = None
        self.time_last_round = None
        self.current_round = 0
        self.tabular = tabular
        self.heading = heading

    def __enter__(self):
        self.start_time = time()
        self.time_last_round = self.start_time
        if self.tabular:
            logging.info("-" * (15 + self.n_digits_step))
            if self.heading:
                logging.info(self.heading)
            logging.info(f"{'Step': <{self.n_digits_step}} | {'Elapsed Time': <12}")
            logging.info("-" * (15 + self.n_digits_step))
        return self
    
    def round(self, round_name):
        time_last_round = self.time_last_round
        self.time_last_round = time()
        time_needed = f"{self.time_last_round - time_last_round:.2f}"
        if self.tabular:
            logging.info(f"{self.current_round+1:>{self.n_digits}}/{self.rounds_total:>{self.n_digits}}: {round_name:<{self.n_digits_step-2*self.n_digits-3}} | {time_needed:>10} s")
        else:
            logging.info(f"Step {self.current_round + 1}/{self.rounds_total} ({round_name}) took {time_needed} s")
        self.current_round += 1
    
    def __exit__(self, exc_type, exc_value, traceback):
        total_time = f"{time() - self.start_time:.2f}"
        if self.tabular:
            logging.info("-" * (15 + self.n_digits_step))
            logging.info(f"{'Total': <{self.n_digits_step}} | {total_time:>10} s")
            logging.info("-" * (15 + self.n_digits_step))
        return False