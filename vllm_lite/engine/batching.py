from collections import deque
from vllm_lite.config import Config

class ScheduledBatch:
    def __init__(self, config:Config):