import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def month_end_draw(
    rng: np.random.Generator,
    start: str,
    end: str,
    n_observations: int,
) -> np.ndarray:
    """Draw random month-end dates from a date range."""
    dates = pd.date_range(start, end, freq="ME")
    return rng.choice(dates.to_numpy(), size=n_observations, replace=True)


def generate_pd_validation_data(
    *,
    seed: int = 20260611,
    n_development: int = 12_000,
    n_validation: int = 6_000,
    n_out_of_time: int = 6_000,
) -> pd.DataFrame:
    """Generate synthetic PD model validation data.
    The function generates explanatory variables, sample labels, exposure,
    and a binary default target.

    """
    rng = np.random.default_rng(seed)

    sample_sizes = {
        "development": n_development,
        "validation": n_validation,
        "out_of_time": n_out_of_time,
    }
    sample_periods = {
        "development": ("2022-01-31", "2023-12-31"),
        "validation": ("2024-01-31", "2024-12-31"),
        "out_of_time": ("2025-01-31", "2025-12-31"),
    }

    segments = np.array(["mortgage", "unsecured", "sme"])
    segment_probs = {
        "development": [0.50, 0.32, 0.18],
        "validation": [0.47, 0.34, 0.19],
        "out_of_time": [0.39, 0.40, 0.21],
    }

    region_names = np.array(["north", "south", "east", "west", "central"])
    region_probs = np.array([0.17, 0.18, 0.16, 0.19, 0.30])

    frames: list[pd.DataFrame] = []

    running_id = 1

    for sample, n_observations in sample_sizes.items():
        is_validation = sample == "validation"
        is_out_of_time = sample == "out_of_time"

        segment = rng.choice(
            segments,
            size=n_observations,
            p=segment_probs[sample],
        )

        region = rng.choice(
            region_names,
            size=n_observations,
            p=region_probs,
        )

        age = (
            np.clip(
                rng.normal(
                    loc=np.where(
                        segment == "mortgage",
                        42,
                        np.where(segment == "unsecured", 36, 45),
                    ),
                    scale=9,
                    size=n_observations,
                ),
                21,
                70,
            )
            .round()
            .astype(int)
        )

        log_income = rng.normal(
            loc=np.where(
                segment == "sme",
                11.9,
                np.where(segment == "mortgage", 11.4, 10.9),
            ),
            scale=np.where(segment == "sme", 0.65, 0.45),
            size=n_observations,
        )
        income = np.exp(log_income)

        debt_to_income = rng.beta(2.2, 5.0, size=n_observations)
        debt_to_income += np.where(segment == "unsecured", 0.08, 0.0)
        debt_to_income += 0.025 if is_validation else 0.0
        debt_to_income += 0.085 if is_out_of_time else 0.0
        debt_to_income = np.clip(debt_to_income, 0.02, 0.98)

        credit_utilization = rng.beta(2.0, 3.2, size=n_observations)
        credit_utilization += np.where(segment == "unsecured", 0.13, 0.0)
        credit_utilization += 0.02 if is_validation else 0.0
        credit_utilization += 0.08 if is_out_of_time else 0.0
        credit_utilization = np.clip(credit_utilization, 0.0, 1.0)

        months_on_book = (
            np.clip(
                rng.gamma(shape=3.0, scale=18.0, size=n_observations)
                + np.where(segment == "mortgage", 12, 0),
                1,
                240,
            )
            .round()
            .astype(int)
        )

        delinquencies_12m = rng.poisson(
            lam=(
                0.10
                + 0.55 * credit_utilization
                + 0.35 * debt_to_income
                + np.where(segment == "unsecured", 0.12, 0.0)
                + (0.10 if is_out_of_time else 0.0)
            ),
            size=n_observations,
        )
        delinquencies_12m = np.clip(delinquencies_12m, 0, 6)

        bureau_score = (
            735
            - 105 * credit_utilization
            - 80 * debt_to_income
            - 24 * delinquencies_12m
            + rng.normal(
                0,
                35 if not is_out_of_time else 44,
                size=n_observations,
            )
            - (12 if is_validation else 0)
            - (30 if is_out_of_time else 0)
        )
        bureau_score = np.clip(bureau_score, 300, 850).round().astype(int)

        loan_to_value = np.where(
            segment == "mortgage",
            rng.beta(4.5, 2.2, size=n_observations),
            np.where(
                segment == "sme",
                rng.beta(3.0, 3.0, size=n_observations),
                np.nan,
            ),
        )
        if is_out_of_time:
            loan_to_value = np.where(
                np.isnan(loan_to_value),
                loan_to_value,
                np.clip(loan_to_value + 0.05, 0.05, 1.30),
            )

        region_effect = (
            pd.Series(region)
            .map(
                {
                    "north": 0.02,
                    "south": 0.12,
                    "east": 0.05,
                    "west": -0.04,
                    "central": 0.00,
                }
            )
            .to_numpy()
        )

        true_logit = (
            -4.30
            + 2.00 * debt_to_income
            + 1.50 * credit_utilization
            + 0.27 * delinquencies_12m
            - 0.006 * (bureau_score - 650)
            - 0.004 * (age - 40)
            + np.where(segment == "unsecured", 0.35, 0.0)
            + np.where(segment == "sme", 0.20, 0.0)
            + np.nan_to_num(0.55 * (loan_to_value - 0.70), nan=0.0)
            + region_effect
            + (0.10 if is_validation else 0.0)
            + (0.42 if is_out_of_time else 0.0)
            + np.where(
                (segment == "unsecured") & is_out_of_time,
                0.18,
                0.0,
            )
            + rng.normal(
                0,
                0.20 if not is_out_of_time else 0.48,
                size=n_observations,
            )
        )
        true_pd = np.clip(sigmoid(true_logit), 0.0005, 0.85)
        default_12m = rng.binomial(1, true_pd, size=n_observations)

        ead = np.where(
            segment == "mortgage",
            rng.lognormal(mean=12.2, sigma=0.55, size=n_observations),
            np.where(
                segment == "unsecured",
                rng.lognormal(mean=10.2, sigma=0.70, size=n_observations),
                rng.lognormal(mean=13.0, sigma=0.85, size=n_observations),
            ),
        )

        income_observed = income.copy()
        bureau_score_observed = bureau_score.astype(float)
        income_missing_probability = 0.015 + (0.020 if is_out_of_time else 0.0)
        bureau_missing_probability = 0.010 + (0.025 if is_out_of_time else 0.0)
        income_observed[rng.random(n_observations) < income_missing_probability] = np.nan
        bureau_score_observed[rng.random(n_observations) < bureau_missing_probability] = np.nan

        frame = pd.DataFrame(
            {
                "observation_id": [
                    f"OBS{idx:07d}" for idx in range(running_id, running_id + n_observations)
                ],
                "observation_date": month_end_draw(
                    rng,
                    *sample_periods[sample],
                    n_observations,
                ),
                "sample": sample,
                "segment": segment,
                "region": region,
                "age": age,
                "annual_income": np.round(income_observed, 2),
                "debt_to_income": np.round(debt_to_income, 4),
                "credit_utilization": np.round(credit_utilization, 4),
                "months_on_book": months_on_book,
                "delinquencies_12m": delinquencies_12m,
                "bureau_score": bureau_score_observed,
                "loan_to_value": np.round(loan_to_value, 4),
                "ead": np.round(ead, 2),
                "default_12m": default_12m,
            }
        )
        frames.append(frame)
        running_id += n_observations

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["observation_date", "observation_id"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    data = generate_pd_validation_data()
    data.to_csv("pd_model_validation_synthetic.csv", index=False)
