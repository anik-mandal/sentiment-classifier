import sys, inspect, transformers
from transformers.training_args import TrainingArguments

print("python exe:", sys.executable)
print("transformers version:", transformers.__version__)
print("transformers file:", transformers.__file__)

sig = inspect.signature(TrainingArguments.__init__)
print("\nTrainingArguments.__init__ signature:\n", sig)
