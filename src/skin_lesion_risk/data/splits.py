"""Patient-level fold helpers.

Concrete implementation should use StratifiedGroupKFold when scikit-learn is
available, with `target` as the stratification label and `patient_id` as group.
"""

