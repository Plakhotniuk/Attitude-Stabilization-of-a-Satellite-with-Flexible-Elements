"""
Exceptions module
"""


class InvalidInputException(Exception):
    """
    Exception class

    Args:
        message (str): error message
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
