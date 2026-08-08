"""External feature sources merged into the player-season frame by (name, season).

Each loader returns a tidy DataFrame keyed on a normalized join key + season and is
merged left onto the nfl_data_py core, so a season with no external file simply
yields NaNs (the models handle missing values natively).
"""
