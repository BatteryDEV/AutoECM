from .EISDataIO import eis_dataframe_from_csv, ECM_from_raw_strings


class TestCircuitFitting:
    eis_data = eis_dataframe_from_csv("train_data.csv")

    def test_l_r_rcpe_rcpe_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "L-R-RCPE-RCPE-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_l_r_rcpe_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "L-R-RCPE-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_l_r_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "L-R-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_rc_g_g(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "RC-G-G"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_rc_rc_rcpe_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "RC-RC-RCPE-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_rcpe_rcpe_rcpe_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "RCPE-RCPE-RCPE-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_rcpe_rcpe_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "RCPE-RCPE-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_rcpe_rcpe(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "RCPE-RCPE"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)

    def test_rs_ws(self):
        sample = self.eis_data.loc[self.eis_data.Circuit == "Rs_Ws"].iloc[0]
        frequencies = sample.freq
        impedances = sample.Z
        circuit = ECM_from_raw_strings(sample.Circuit, sample.Parameters)
        params_before_fit = circuit.param_values
        circuit.fit(frequencies, impedances)
        params_after_fit = circuit.param_values
        assert params_after_fit != params_before_fit
        assert len(params_after_fit) == len(params_before_fit)
