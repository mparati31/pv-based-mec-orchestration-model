FILE NAMING

The name of a file directly reflects its content as described in the paper (Section 5.1, pag. 14).

For datasets A and B each instance is stored in a distinct file, whose name contains the number of time-slots |T| and the instance number: 

dataset_{dataset label}_{number of time-slots}t.dat

For dataset C  each instance is stored in a distinct file, whose name contains the number of hours used for the merging:

dataset_C_{number of hours}h.dat

while the file "dataset_C_insa-cl.dat" contains the data merged following the ad-hoc clustering.

FILE FORMAT

all files share the following format:

- 1 integer: number of nodes
- 1 integer: number of facilities
- 1 integer: number of time-slots
- 1 double: alpha parameter
- 1 double: beta parameter
- 1 double: U parameter
- 1 double > 1: maximum capacity of facility in absolute value
- matrix A x T of doubles, comma separated values: demand of node in time
- matrix K x K of doubles, comma separated values: distance between pair of facilities
- matrix A x K of doubles, comma separated values: distance between nodes and facilities


