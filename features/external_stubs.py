"""Stub external feature sources — ready to implement when data sources are available.

These use the ExternalFeatureSource base class which fetches data from external
APIs/websites into data/external/<name>/ before building features.

To implement a stub:
1. Fill in fetch() to download data to self.external_data_dir(data_dir)
2. Fill in build() to read from that directory and return a DataFrame
3. Enable the feature name in config.py ENABLED_FEATURES

Implemented sources (moved to their own modules):
- VegasOddsFeatures -> features/vegas.py
- RosterContinuityFeatures -> features/roster.py
- KenPomFeatures, APPollFeatures, PublicPicksFeatures -> features/kenpom.py
"""
