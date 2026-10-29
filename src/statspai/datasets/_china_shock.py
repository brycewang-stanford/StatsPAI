"""Autor-Dorn-Hanson (2013) China-shock commuting-zone panel, fetched on demand.

The ``workfile_china.dta`` of Autor, Dorn and Hanson's "The China
Syndrome" [@autor2013china]: 722 commuting zones in two decades
(1990-2000 and 2000-2007), 1,444 rows. StatsPAI fetches it from the
replication package of Agarwal and Singh [@agarwal2026replication]
(Zenodo record 19866751, 9.6 MB), whose ``data/LICENSE.txt`` puts the
data under CC-BY-4.0 (copyright 2013 American Economic Association). The
archive is checked against a pinned SHA-256 and cached
(:mod:`statspai.datasets._fetch`); nothing is redistributed.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from ._fetch import RemoteFile, fetch

AGARWAL_SINGH_PACKAGE = RemoteFile(
    filename="agarwal_singh_2026_replication_package.zip",
    url="https://zenodo.org/records/19866751/files/replication_package.zip"
    "?download=1",
    sha256="92d2df42204f38e9e70d077bfa01b514ab102f9c4614b12a1761fe4a0e633348",
    size=9615512,
    licence="CC-BY-4.0 (Zenodo record 19866751); the ADH workfile inside is "
    "(c) 2013 American Economic Association, CC-BY-4.0; fetched, not "
    "redistributed",
    origin="Agarwal and Singh (2026) Econometrica replication package, "
    "doi:10.5281/zenodo.19866751",
)

#: Member of the archive that holds the ADH workfile.
DTA_MEMBER = "replication_package/data/workfile_china.dta"

#: Columns of the frame ``china_shock()`` returns (listed without downloading).
COLUMNS = (
    "czone",
    "statefip",
    "city",
    "yr",
    "t2",
    "timepwt48",
    "reg_midatl",
    "reg_encen",
    "reg_wncen",
    "reg_satl",
    "reg_escen",
    "reg_wscen",
    "reg_mount",
    "reg_pacif",
    "l_popcount",
    "l_no_workers_totcbp",
    "l_shind_manuf_cbp",
    "l_sh_popedu_c",
    "l_sh_popfborn",
    "l_sh_empl_f",
    "l_sh_routine33",
    "l_task_outsource",
    "l_sh_empl",
    "l_sh_empl_mfg",
    "l_sh_empl_mfg_m",
    "l_sh_empl_mfg_f",
    "l_sh_empl_mfg_edu_nc",
    "l_sh_empl_mfg_edu_c",
    "l_sh_empl_mfg_age1634",
    "l_sh_empl_mfg_age3549",
    "l_sh_empl_mfg_age5064",
    "l_sh_empl_nmfg",
    "l_sh_empl_nmfg_m",
    "l_sh_empl_nmfg_f",
    "l_sh_empl_nmfg_edu_nc",
    "l_sh_empl_nmfg_edu_c",
    "l_sh_empl_nmfg_age1634",
    "l_sh_empl_nmfg_age3549",
    "l_sh_empl_nmfg_age5064",
    "l_sh_unempl",
    "l_sh_unempl_m",
    "l_sh_unempl_f",
    "l_sh_unempl_edu_nc",
    "l_sh_unempl_edu_c",
    "l_sh_unempl_age1634",
    "l_sh_unempl_age3549",
    "l_sh_unempl_age5064",
    "l_sh_nilf",
    "l_sh_nilf_m",
    "l_sh_nilf_f",
    "l_sh_nilf_edu_nc",
    "l_sh_nilf_edu_c",
    "l_sh_nilf_age1634",
    "l_sh_nilf_age3549",
    "l_sh_nilf_age5064",
    "l_sh_ssadiswkrs",
    "l_avg_lnwkwage_mfg",
    "l_avg_lnwkwage_nmfg",
    "l_trans_totindiv_pc",
    "l_trans_totmed_pc",
    "l_trans_fedinc_pc",
    "l_trans_othinc_pc",
    "l_trans_unemp_pc",
    "l_trans_taaimp_pc",
    "l_trans_totedu_pc",
    "l_trans_ssaret_pc",
    "l_trans_ssadis_pc",
    "l_avg_hhincsum_pc_pw",
    "l_avg_hhincwage_pc_pw",
    "l_tradeusch_pw",
    "l_tradeotch_pw",
    "d_tradeusch_pw",
    "d_tradeotch_pw_lag",
    "lnchg_popworkage",
    "lnchg_popworkage_edu_c",
    "lnchg_popworkage_edu_nc",
    "lnchg_popworkage_age1634",
    "lnchg_popworkage_age3549",
    "lnchg_popworkage_age5064",
    "d_popcount",
    "d_sh_empl",
    "d_sh_empl_mfg",
    "d_sh_empl_nmfg",
    "d_sh_empl_mfg_m",
    "d_sh_empl_nmfg_m",
    "d_sh_empl_mfg_f",
    "d_sh_empl_nmfg_f",
    "d_sh_empl_mfg_edu_nc",
    "d_sh_empl_nmfg_edu_nc",
    "d_sh_empl_mfg_edu_c",
    "d_sh_empl_nmfg_edu_c",
    "d_sh_empl_mfg_age1634",
    "d_sh_empl_nmfg_age1634",
    "d_sh_empl_mfg_age3549",
    "d_sh_empl_nmfg_age3549",
    "d_sh_empl_mfg_age5064",
    "d_sh_empl_nmfg_age5064",
    "d_sh_empl_edu_c",
    "d_sh_empl_edu_nc",
    "d_sh_unempl",
    "d_sh_unempl_m",
    "d_sh_unempl_f",
    "d_sh_unempl_edu_nc",
    "d_sh_unempl_edu_c",
    "d_sh_unempl_age1634",
    "d_sh_unempl_age3549",
    "d_sh_unempl_age5064",
    "d_sh_nilf",
    "d_sh_nilf_m",
    "d_sh_nilf_f",
    "d_sh_nilf_edu_nc",
    "d_sh_nilf_edu_c",
    "d_sh_nilf_age1634",
    "d_sh_nilf_age3549",
    "d_sh_nilf_age5064",
    "d_sh_ssadiswkrs",
    "d_avg_lnwkwage",
    "d_avg_lnwkwage_c",
    "d_avg_lnwkwage_nc",
    "d_avg_lnwkwage_m",
    "d_avg_lnwkwage_f",
    "d_avg_lnwkwage_c_m",
    "d_avg_lnwkwage_c_f",
    "d_avg_lnwkwage_nc_m",
    "d_avg_lnwkwage_nc_f",
    "d_avg_lnwkwage_mfg",
    "d_avg_lnwkwage_mfg_c",
    "d_avg_lnwkwage_mfg_nc",
    "d_avg_lnwkwage_nmfg",
    "d_avg_lnwkwage_nmfg_c",
    "d_avg_lnwkwage_nmfg_nc",
    "lnchg_no_empl",
    "lnchg_no_empl_mfg",
    "lnchg_no_empl_nmfg",
    "lnchg_no_empl_mfg_edu_nc",
    "lnchg_no_empl_nmfg_edu_nc",
    "lnchg_no_empl_mfg_edu_c",
    "lnchg_no_empl_nmfg_edu_c",
    "lnchg_no_unempl",
    "lnchg_no_nilf",
    "lnchg_no_ssadiswkrs",
    "lnchg_trans_totindiv_pc",
    "lnchg_trans_totmed_pc",
    "lnchg_trans_fedinc_pc",
    "lnchg_trans_othinc_pc",
    "lnchg_trans_unemp_pc",
    "lnchg_trans_taaimp_pc",
    "lnchg_trans_totedu_pc",
    "lnchg_trans_ssaret_pc",
    "lnchg_trans_ssadis_pc",
    "d_trans_totindiv_pc",
    "d_trans_totmed_pc",
    "d_trans_fedinc_pc",
    "d_trans_othinc_pc",
    "d_trans_unemp_pc",
    "d_trans_taaimp_pc",
    "d_trans_totedu_pc",
    "d_trans_ssaret_pc",
    "d_trans_ssadis_pc",
    "relchg_avg_hhincsum_pc_pw",
    "relchg_avg_hhincwage_pc_pw",
    "relchg_avg_hhincbusinv_pc_pw",
    "relchg_avg_hhinctrans_pc_pw",
    "relchg_avg_hhincrest_pc_pw",
    "relchg_med_hhincsum_pc_pw",
    "relchg_med_hhincwage_pc_pw",
    "d_avg_hhincsum_pc_pw",
    "d_avg_hhincwage_pc_pw",
    "d_avg_hhincbusinv_pc_pw",
    "d_avg_hhinctrans_pc_pw",
    "d_avg_hhincrest_pc_pw",
    "d_med_hhincsum_pc_pw",
    "d_med_hhincwage_pc_pw",
    "d_tradex_usch_pw",
    "d_tradex_otch_pw_lag",
    "d_tradeusch_netinput_pw",
    "d_inputotch_pw_lag",
    "d_netimpusch_pw",
    "d_expotch_pw_lag",
    "d_traderes_pw_lag",
    "d_nettradefactor_usch_io",
    "d_tradefactor_otch_lag_io",
    "d_expfactor_otch_lag_io",
    "d_tradeuschlw_pw",
    "d_tradeotchlw_pw_lag",
    "d_tradeuschce_pw",
    "d_tradeotchce_pw_lag",
    "d_tradeusce_pw",
    "d_tradeotce_pw_lag",
    "d_tradeushi_pw",
    "d_tradeothi_pw_lag",
)


def china_shock(download: bool = True) -> pd.DataFrame:
    """Autor-Dorn-Hanson (2013) China-shock panel — real data.

    The commuting-zone workfile behind "The China Syndrome" by Autor, Dorn
    and Hanson [@autor2013china]. It holds changes in manufacturing
    employment, unemployment, wages and transfers against the change in
    Chinese import exposure per worker, for 722 commuting zones over
    1990-2000 and 2000-2007. The file is fetched on first use from the
    replication package of Agarwal and Singh [@agarwal2026replication],
    which uses it for the noisy-covariate application of
    ``sp.noisy_covariates``, and cached; see
    :mod:`statspai.datasets._fetch`.

    Parameters
    ----------
    download : bool, default True
        Download the archive when it is not cached. With ``False`` a
        missing file raises instead of touching the network.

    Returns
    -------
    pd.DataFrame
        The 191 columns of ``workfile_china.dta`` in file order, 1,444
        rows. Key variables: ``czone`` (commuting zone), ``yr`` (1990 or
        2000, start of the decade), ``t2`` (second decade), ``statefip``,
        ``timepwt48`` (start-of-period population weight), the outcome
        ``d_sh_empl_mfg`` (change in the manufacturing employment share of
        the working-age population), the exposure ``d_tradeusch_pw`` (US
        imports from China per worker), its instrument
        ``d_tradeotch_pw_lag`` (other high-income countries' imports from
        China), the census-division dummies ``reg_*`` and the
        start-of-period controls ``l_*``. Stata's single-precision columns
        are widened to float64 (their values are the float32 values, not
        re-rounded decimals). ``attrs`` carries the source URL, the pinned
        SHA-256 and the licence.

    Raises
    ------
    DataInsufficient
        Offline and not cached, or the download does not match the pinned
        SHA-256. The message gives the URL and the path to save it at.

    Notes
    -----
    With ``estimand="partially_linear_iv"``, the instrument above,
    population weights, ``rank=5`` and the 31 covariates of Agarwal and
    Singh's augmented specification, ``sp.noisy_covariates`` reproduces
    their clean-data estimate -0.61799 (SE 0.06331) with their fold
    assignment
    (``tests/reference_parity/test_noisy_covariates_authors_parity.py``).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.china_shock()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (1444, 191)
    >>> fit = sp.ivreg("d_sh_empl_mfg ~ t2 + (d_tradeusch_pw ~ "
    ...                "d_tradeotch_pw_lag)", data=df)  # doctest: +SKIP

    References
    ----------
    [@autor2013china]

    [@agarwal2026replication]
    """
    path = fetch(AGARWAL_SINGH_PACKAGE, download=download)
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        raw = archive.read(DTA_MEMBER)
    df = pd.read_stata(io.BytesIO(raw))
    single = [c for c in df.columns if df[c].dtype == np.float32]
    df[single] = df[single].astype("float64")
    df.attrs["paper"] = (
        "Autor, Dorn and Hanson (2013), The China Syndrome: Local Labor "
        "Market Effects of Import Competition in the United States, AER "
        "103(6)"
    )
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_url"] = AGARWAL_SINGH_PACKAGE.url
    df.attrs["sha256"] = AGARWAL_SINGH_PACKAGE.sha256
    df.attrs["licence"] = AGARWAL_SINGH_PACKAGE.licence
    df.attrs["source_origin"] = AGARWAL_SINGH_PACKAGE.origin
    return df
