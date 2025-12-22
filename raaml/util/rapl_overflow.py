class RAPLOverflowError(Exception):
    """Exception raised when RAPL counter overflow is detected."""

    def __init__(self, message="RAPL counter overflow detected"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message
