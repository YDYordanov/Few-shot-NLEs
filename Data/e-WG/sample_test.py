"""
Randomly sample 100 instances from WG-dev as test set.
These instances will be used to generate human and model NLEs for them,
and be used for final hand-evaluation via the NLE score.
Note: we use the full test set instead for model accuracy reporting.
"""

import random
import jsonlines


data_path = 'dev.jsonl'
with jsonlines.open(data_path) as reader:
    data = list(reader)
    
random.seed(4509)

test_size = 100
test_set = random.sample(data, test_size)

# Write the test data to file
out_file = 'test_100.jsonl'
with jsonlines.open(out_file, 'w') as writer:
    writer.write_all(test_set)

