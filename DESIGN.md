# Design notes

This document explains *why* `systile` is shaped the way it is. The short version:
a TPU does not see a flat array, so a data structure built for a TPU should not
pretend it does.

