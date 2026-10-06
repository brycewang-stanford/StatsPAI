# Draft note to Klaus Neusser on `KalmanSmootherTVP.m`

Status: final text, to be sent by Bryce (decision delegated 2026-10-06:
send). Not sent by the assistant.

---

Subject: A transpose in KalmanSmootherTVP.m (Time Series Econometrics, section 17.4)

Dear Professor Neusser,

I maintain StatsPAI, an open-source Python package for econometrics and
causal inference (https://github.com/brycewang-stanford/StatsPAI). We
have been working through the examples of *Time Series Econometrics* to
check the package's time-series functions against your data, and the
mixed-frequency model of section 17.4 (quarterly GDP from annual data) is
now one of its test cases. While checking its Kalman filter and smoother
against the MATLAB code you provide for that section, I found what looks
like a slip in the smoother. I am writing in case it is useful for the
errata.

In `KalmanSmootherTVP.m` the smoothed mean is updated with

    XT(:,t-1) = Xt(:,t-1) + Pt(:,:,t-1)*F(:,:,t)'*inv(Ptp1)*(XT(:,t)-F(:,:,t)*Xt(:,t-1));

which is the usual fixed-interval recursion with gain
`J = Pt*F'*inv(Ptp1)`. The next line updates the variance with

    PT(:,:,t-1) = Pt(:,:,t-1) + Pt(:,:,t-1)*F(:,:,t)*inv(Ptp1)*(PT(:,:,t)-Ptp1)*inv(Ptp1)*Pt(:,:,t-1);

Here `F` appears where `F'` is needed on the left, and the factor `F` is
missing on the right. With the same gain the line would read

    PT(:,:,t-1) = Pt(:,:,t-1) + Pt(:,:,t-1)*F(:,:,t)'*inv(Ptp1)*(PT(:,:,t)-Ptp1)*inv(Ptp1)*F(:,:,t)*Pt(:,:,t-1);

that is, `P(t-1|T) = P(t-1|t-1) + J (P(t|T) - P(t|t-1)) J'`.

What I checked, on the data file shipped with the code and at the
maximum-likelihood estimates:

- The filter (`KalmanFilterTVP.m`) is fine. A line-by-line port agrees
  with StatsPAI's filter (`sp.kalman_filter`, written independently) to
  7e-16, and its log-likelihood
  differs from one that skips the missing annual observations by exactly
  `0.5*log(2*pi)` per missing value, as it should.
- The smoothed means are not affected by the variance line; they agree
  with StatsPAI's smoother to 1e-15.
- The smoothed variances are affected. For the quarterly growth state,
  away from the last four quarters, the code as shipped gives variances
  between 0.044 and 0.057; the corrected recursion gives 0.029 to 0.048.
  (At the last date the two coincide at 0.087, the filtered variance.)
  The largest difference over all elements and dates is 0.134. The
  corrected line agrees with StatsPAI's smoother to 2e-16, and that
  smoother agrees with the R package KFAS on six other models to 5e-15.

The consequence is limited to the second figure produced by `main.m`: the
confidence band around the smoothed quarterly growth rate is drawn wider
than it should be. The point estimates and everything in the first figure
are unchanged.

I have not seen the printed figure in the book next to these numbers, so I
cannot say whether it was produced with this file.

The comparison is a test in the package's repository
(`tests/external_parity/test_neusser_quarterly_gdp.py`); it needs your
data file, which we do not redistribute. Thank you for making the data
and code of the book available.

With best regards,
Bryce Wang
