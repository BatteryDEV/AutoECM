from tempfile import NamedTemporaryFile
from numpy import nan
from .scoring import (
    circuit_CV_optimized_score,
    circuit_rmse_score,
    ECM_class_classification_scorer,
    ECM_params_initial_guess_scorer,
)
from .EISDataIO import ECM_from_raw_strings, eis_dataframe_from_csv, eis_dataframe_to_csv


class Test_EIS_IO_scoring:
    train_data = eis_dataframe_from_csv("train_data.csv")
    submission_data = eis_dataframe_from_csv("submission_example.csv")

    def test_classification_score(self):
        actual = self.train_data["Circuit"]
        predicted_ish = self.train_data["Circuit"].sample(frac=1.0)
        score = ECM_class_classification_scorer(actual, predicted_ish)
        assert isinstance(score, float)
        assert 0 <= score
        assert 1 >= score

    def test_parameters_guess_score(self):
        num_samples = 500
        actual_eis_data = self.train_data[["freq", "Z"]].loc[:num_samples, :]
        predicted_ish = self.train_data[["Circuit", "Parameters"]].sample(n=num_samples)
        score = ECM_params_initial_guess_scorer(actual_eis_data, predicted_ish)
        assert isinstance(score, float)
        assert 0 <= score

    def test_manual_parse_and_score_optimized_CV_scores_from_submission(self):
        submissions_w_params = self.submission_data.dropna(axis=0).sample(n=100)
        for index in submissions_w_params.index.values:
            sample = self.submission_data.loc[index, :]
            real = self.train_data.loc[index, :]
            try:
                ecm = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
                score = circuit_CV_optimized_score(ecm, real.freq, real.Z)
                assert isinstance(score, float)
            except AssertionError:
                raise AssertionError(
                    f"Unable to parse and score submission with circuit {sample.Circuit} from row {index}"
                )

    def test_manual_parse_and_score_rmse_scores_from_submission(self):
        submissions_w_params = self.submission_data.dropna(axis=0)
        for index in submissions_w_params.index.values:
            sample = self.submission_data.loc[index, :]
            real = self.train_data.loc[index, :]
            try:
                ecm = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
                score = circuit_rmse_score(ecm, real.freq, real.Z)
                assert isinstance(score, float)
            except AssertionError:
                raise AssertionError(
                    f"Unable to parse and score submission with circuit {sample.Circuit} from row {index}"
                )

    def test_write_and_read_submission(self):
        subset = self.train_data.loc[:, ["Circuit", "Parameters"]]
        # randomly delete 20% of parameters to simulate a contestant not guessing all of them
        subset.loc[subset.sample(frac=0.2).index, "Parameters"] = nan
        with NamedTemporaryFile("w") as temp:
            eis_dataframe_to_csv(subset, temp.name)
            read_it_back = eis_dataframe_from_csv(temp.name)
            assert read_it_back.equals(subset)

