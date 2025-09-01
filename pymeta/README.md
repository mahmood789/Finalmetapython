# pymeta

A modular meta-analysis platform with specialized engines for bias correction, Bayesian stacking, ML-based heterogeneity, trial simulation, network/dose-response, living reviews, NLP extraction, protocol generation, validation, and FAIR export.

## Structure

- `app01_meta_engine`: Core meta-analysis
- `app02_bias_sensitivity`: Bias corrections & sensitivity
- `app03_bayesian_engine`: Bayesian meta-analysis & stacking
- `app04_transport_weights`: Transportability & weighting
- `app05_ml_heterogeneity`: ML models & SHAP
- `app06_trial_sims`: MAMS, Platform, Basket simulators
- `app07_network_dose`: Network & dose-response meta-analysis
- `app08_living_auto`: Living systematic review automation
- `app09_nlp_extract`: Abstract & full-text PDF extractor
- `app10_protocol_builder`: SPIRIT/CONSORT protocol generator
- `app11_validation`: CONSORT, GRADE, FDA/EMA validation
- `app12_fair_export`: FAIR export, Zenodo/OSF packaging
- `tests`: Test files per app and full pipeline
- `docker`: Containerization per app
- `.github/workflows`: CI/CD configs

## Requirements

See `requirements.txt`.